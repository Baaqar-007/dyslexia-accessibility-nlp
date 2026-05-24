"""
app/main.py

Production-grade Flask application.

Fixes over original:
  - Models were reloaded from disk on every request. Now loaded once at
    startup via pipeline/inference.py's @lru_cache functions.
  - No input validation — any file (or no file) was accepted silently.
  - Port was 501 (non-standard). Fixed to 5000 via config.
  - shutil.rmtree in a finally-block deleted ALL output on error.
    Now only the session-specific temp directory is cleaned up per request.
  - PDF was written to a fixed path, meaning concurrent requests overwrote
    each other. Now each session gets a unique UUID-keyed filename.
  - No MIME-type check — only file extension was checked.
  - No request size limit enforced at the route level.
  - Flask debug=True was the default, exposing the interactive debugger
    in what could easily end up as a production deployment.
  - All logic was in a single 350-line file with no separation of concerns.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask, jsonify, request, send_file,
    render_template, abort,
)
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Detect GPU once at startup ────────────────────────────────────────────────
from gpu_config import configure_gpu
configure_gpu()

from config import AppConfig, Paths

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = AppConfig.MAX_CONTENT_LENGTH

# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
Paths.CHAR_DIR.mkdir(parents=True, exist_ok=True)
Paths.REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy model preload (runs in background thread to keep startup fast)
# ---------------------------------------------------------------------------
def _preload_in_background():
    try:
        from pipeline.inference import preload_models
        preload_models()
    except Exception as exc:
        logger.warning("Background model preload failed: %s", exc)

threading.Thread(target=_preload_in_background, daemon=True).start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lstrip(".").lower()
    return suffix in AppConfig.ALLOWED_EXTENSIONS


def _validate_mime(file_storage) -> bool:
    """Read the first 12 bytes to verify common image magic bytes."""
    header = file_storage.stream.read(12)
    file_storage.stream.seek(0)
    return (
        header[:4]  in {b"\x89PNG", b"\xff\xd8\xff\xe0",
                         b"\xff\xd8\xff\xe1", b"\xff\xd8\xff\xdb"} or
        header[:2]  == b"BM" or          # BMP
        header[:4]  == b"II\x2a\x00" or  # TIFF little-endian
        header[:4]  == b"MM\x00\x2a"     # TIFF big-endian
    )


def _schedule_deletion(path: str, delay: int = AppConfig.REPORT_EXPIRY_SEC):
    """Delete a file after `delay` seconds in a daemon thread."""
    def _delete():
        time.sleep(delay)
        try:
            os.remove(path)
            logger.debug("Auto-deleted: %s", path)
        except FileNotFoundError:
            pass
    threading.Thread(target=_delete, daemon=True).start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload
    Accepts: multipart/form-data with field 'file'
    Returns: JSON DiagnosisResult + pdf_url
    """
    # ---- Validate presence --------------------------------------------------
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected."}), 400

    # ---- Validate extension -------------------------------------------------
    if not _allowed_file(f.filename):
        return jsonify({
            "error": (
                f"Unsupported file type. Allowed: "
                f"{', '.join(sorted(AppConfig.ALLOWED_EXTENSIONS))}"
            )
        }), 415

    # ---- Validate MIME (magic bytes) ----------------------------------------
    if not _validate_mime(f):
        return jsonify({"error": "File content does not match an image format."}), 415

    # ---- Save upload to a session-scoped temp directory --------------------
    session_id   = uuid.uuid4().hex
    session_dir  = Paths.CHAR_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    safe_name    = secure_filename(f.filename)
    upload_path  = str(session_dir / safe_name)
    f.save(upload_path)
    logger.info("Upload saved: %s  (session=%s)", safe_name, session_id)

    try:
        # ---- Inference ------------------------------------------------------
        from pipeline.inference import run_inference
        diagnosis = run_inference(upload_path)

        # ---- Generate PDF ---------------------------------------------------
        from pipeline.report_generator import generate_report
        pdf_filename = f"report_{session_id}.pdf"
        pdf_path     = generate_report(
            diagnosis,
            image_path=upload_path,
            filename=pdf_filename,
        )
        # Auto-delete PDF after expiry
        _schedule_deletion(pdf_path)

        # ---- Build response payload -----------------------------------------
        payload = diagnosis.to_dict()
        payload["pdf_url"]  = f"/download_report/{pdf_filename}"
        payload["session"]  = session_id

        return jsonify(payload), 200

    except Exception as exc:
        logger.exception("Inference error for session %s", session_id)
        return jsonify({
            "error":   "Internal analysis error.",
            "detail":  str(exc),
            "session": session_id,
        }), 500

    finally:
        # Clean up only the session's temp character crops, not the report
        import shutil
        try:
            shutil.rmtree(str(session_dir), ignore_errors=True)
        except Exception:
            pass


@app.route("/download_report/<filename>")
def download_report(filename: str):
    """
    GET /download_report/<filename>
    Serves the PDF report.  Filename is UUID-keyed — no path traversal possible.
    """
    # Guard: only allow filenames that match our naming convention
    safe = secure_filename(filename)
    if safe != filename or not filename.startswith("report_") or not filename.endswith(".pdf"):
        abort(400)

    pdf_path = Paths.REPORTS / safe
    if not pdf_path.exists():
        abort(404)

    return send_file(
        str(pdf_path),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=safe,
    )


@app.route("/health")
def health():
    """Simple health-check endpoint for container orchestration."""
    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request.", "detail": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found."}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": f"File too large. Maximum size: "
                 f"{AppConfig.MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
    }), 413

@app.errorhandler(415)
def unsupported_media(e):
    return jsonify({"error": "Unsupported media type."}), 415

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        debug=AppConfig.DEBUG,
    )
