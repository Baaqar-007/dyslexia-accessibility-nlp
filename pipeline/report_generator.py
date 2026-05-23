"""
pipeline/report_generator.py

Generates a structured professional PDF report from a DiagnosisResult.

Improvements over original:
  - Original used the low-level ReportLab Canvas API, hand-positioning every
    string.  This module uses ReportLab Platypus (document templates +
    flowables), which handles pagination, spacing, and reflowing automatically.
  - Original report had no per-character confidence table.
  - Original had no score breakdown or confidence indicators.
  - Original had no recommendations section.
  - Original embedded the NLP result as a hardcoded 'sample_sequence' string.
  - Added colour-coded summary banner (green / amber / red).
  - Added score bar visualisation drawn with ReportLab graphics.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Paths, EnsembleConfig

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_bar(score: float, width: float = 12 * cm, height: float = 0.6 * cm) -> Drawing:
    """Draw a horizontal score bar: green (left) → red (right), with marker."""
    d = Drawing(width, height)
    # Background track
    d.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#e8e8e8"), strokeColor=None))
    # Filled portion
    filled_w = width * score
    fill_col = (
        colors.HexColor("#27ae60") if score < 0.35 else
        colors.HexColor("#e67e22") if score < 0.55 else
        colors.HexColor("#e74c3c")
    )
    d.add(Rect(0, 0, filled_w, height, fillColor=fill_col, strokeColor=None))
    # Threshold line
    threshold_x = width * EnsembleConfig.DYSLEXIA_THRESHOLD
    d.add(Rect(threshold_x - 1, -2, 2, height + 4,
               fillColor=colors.HexColor("#2c3e50"), strokeColor=None))
    return d


def _result_colour(result: str) -> colors.Color:
    return (
        colors.HexColor("#27ae60") if result == "No Dyslexia Detected"  else
        colors.HexColor("#e74c3c") if result == "Dyslexia Detected"     else
        colors.HexColor("#e67e22")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    diagnosis,                      # DiagnosisResult
    image_path: Optional[str] = None,
    out_dir:    Path           = Paths.REPORTS,
    filename:   str            = "dyslexia_report.pdf",
) -> str:
    """
    Build and save the PDF report.

    Parameters
    ----------
    diagnosis  : DiagnosisResult from pipeline/inference.py
    image_path : optional path to the uploaded handwriting image to embed
    out_dir    : output directory
    filename   : output filename

    Returns
    -------
    Absolute path to the saved PDF (str)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = str(out_dir / filename)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title="Dyslexia Detection Report",
        author="Dyslexia Accessibility NLP",
    )

    styles = getSampleStyleSheet()
    story  = []

    # ---- Title ---------------------------------------------------------------
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=22,
        spaceAfter=4,
        textColor=colors.HexColor("#2c3e50"),
    )
    sub_style = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=2,
    )
    story.append(Paragraph("Dyslexia Detection Report", title_style))
    story.append(Paragraph("Comprehensive Handwriting Analysis", sub_style))
    now = datetime.datetime.now().strftime("%d %B %Y  %H:%M:%S")
    story.append(Paragraph(f"Generated: {now}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#2c3e50"), spaceAfter=12))

    # ---- Diagnosis banner ----------------------------------------------------
    banner_colour = _result_colour(diagnosis.result)
    banner_style  = ParagraphStyle(
        "Banner",
        fontSize=16,
        textColor=colors.white,
        backColor=banner_colour,
        borderPadding=(8, 12, 8, 12),
        spaceAfter=12,
        leading=20,
    )
    confidence_icon = {"High": "●●●", "Medium": "●●○", "Low": "●○○"}.get(
        diagnosis.confidence_label, "●○○"
    )
    story.append(Paragraph(
        f"{diagnosis.result}  |  Confidence: {diagnosis.confidence_label} {confidence_icon}",
        banner_style,
    ))

    # ---- Score breakdown table -----------------------------------------------
    h3 = ParagraphStyle("H3", parent=styles["Heading3"],
                        textColor=colors.HexColor("#2c3e50"), spaceAfter=6)
    story.append(Paragraph("Score Breakdown", h3))

    score_data = [
        ["Component", "Score", "Weight", "Contribution"],
        [
            "CNN Reversal Rate",
            f"{diagnosis.reversal_rate:.1%}",
            "50 %",
            f"{diagnosis.reversal_rate * 0.50:.1%}",
        ],
        [
            "NLP Sequence Anomaly",
            f"{diagnosis.nlp_anomaly_score:.1%}",
            "35 %",
            f"{diagnosis.nlp_anomaly_score * 0.35:.1%}",
        ],
        [
            "MLP Letter Uncertainty",
            f"{diagnosis.mlp_uncertainty:.1%}",
            "15 %",
            f"{diagnosis.mlp_uncertainty * 0.15:.1%}",
        ],
        ["", "", "Ensemble Score →", f"{diagnosis.ensemble_score:.1%}"],
    ]

    score_table = Table(score_data, colWidths=[6 * cm, 3 * cm, 3 * cm, 3.5 * cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("FONTNAME",    (0, -1), (-1, -1), "Helvetica-Bold"),
        ("BACKGROUND",  (0, -1), (-1, -1), colors.HexColor("#ecf0f1")),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 6))

    # Score bar
    story.append(Paragraph("Ensemble score (▐ = dyslexia threshold):", sub_style))
    story.append(_score_bar(diagnosis.ensemble_score))
    story.append(Spacer(1, 14))

    # ---- Detected sequence & character count ---------------------------------
    story.append(Paragraph("Analysis Summary", h3))
    summary_data = [
        ["Characters detected", str(diagnosis.num_characters)],
        ["Predicted sequence",  diagnosis.predicted_sequence or "(none)"],
        ["Reversal count",
         str(sum(1 for c in diagnosis.per_character if c.is_reversal))],
    ]
    sum_table = Table(summary_data, colWidths=[7 * cm, 8.5 * cm])
    sum_table.setStyle(TableStyle([
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f8f9fa"), colors.white]),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(sum_table)
    story.append(Spacer(1, 14))

    # ---- Per-character table -------------------------------------------------
    if diagnosis.per_character:
        story.append(Paragraph("Per-Character Analysis", h3))
        char_header = ["#", "Letter", "MLP Confidence", "Reversal Prob.", "Reversal?"]
        char_rows = [char_header]
        for c in diagnosis.per_character:
            char_rows.append([
                str(c.index + 1),
                c.letter,
                f"{c.mlp_confidence:.1%}",
                f"{c.reversal_prob:.1%}",
                "YES" if c.is_reversal else "no",
            ])
        char_table = Table(char_rows,
                           colWidths=[1.2 * cm, 2.2 * cm, 4 * cm, 4 * cm, 3 * cm])
        row_bg = []
        for row_idx, c in enumerate(diagnosis.per_character, start=1):
            if c.is_reversal:
                row_bg.append(("BACKGROUND", (0, row_idx), (-1, row_idx),
                                colors.HexColor("#fdecea")))
        char_style = TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f8f9fa"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#dee2e6")),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ] + row_bg)
        char_table.setStyle(char_style)
        story.append(char_table)
        story.append(Spacer(1, 14))

    # ---- Recommendations -----------------------------------------------------
    story.append(Paragraph("Recommendations", h3))
    if diagnosis.result == "Dyslexia Detected":
        recs = [
            "Refer the individual to an educational psychologist or specialist for formal assessment.",
            "Consider structured literacy interventions (e.g. Orton-Gillingham approach).",
            "Provide assistive technology such as text-to-speech software.",
            "Allow extended time on written tasks and examinations.",
            "This report is a screening aid only and does not constitute a clinical diagnosis.",
        ]
    elif diagnosis.result == "No Dyslexia Detected":
        recs = [
            "No significant dyslexic patterns detected in this sample.",
            "Continue monitoring handwriting development, particularly letter formation.",
            "If concerns persist, a formal educational assessment is still advisable.",
            "This report is a screening aid only and does not constitute a clinical diagnosis.",
        ]
    else:
        recs = [
            "The analysis was inconclusive — the image may have insufficient characters or quality.",
            "Please upload a clearer handwriting sample with multiple distinct characters.",
            "If you have clinical concerns, consult an educational psychologist.",
        ]

    for rec in recs:
        story.append(Paragraph(f"• {rec}",
                               ParagraphStyle("Rec", parent=styles["Normal"],
                                              fontSize=9, leftIndent=12,
                                              spaceAfter=4)))
    story.append(Spacer(1, 14))

    # ---- Uploaded image ------------------------------------------------------
    if image_path and Path(image_path).exists():
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#bdc3c7")))
        story.append(Spacer(1, 8))
        story.append(Paragraph("Uploaded Handwriting Sample", h3))
        max_img_w = PAGE_W - 2 * MARGIN
        story.append(Image(image_path, width=max_img_w,
                           height=min(10 * cm, max_img_w * 0.5),
                           kind="proportional"))

    # ---- Footer disclaimer ---------------------------------------------------
    story.append(Spacer(1, 20))
    footer_style = ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=7,
        textColor=colors.HexColor("#95a5a6"), alignment=1,
    )
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#dee2e6")))
    story.append(Paragraph(
        "This report is generated by an automated AI screening tool. "
        "It is intended for educational and research purposes only. "
        "It does not constitute a medical or psychological diagnosis. "
        "Always consult a qualified professional for clinical assessment.",
        footer_style,
    ))

    doc.build(story)
    logger.info("Report saved → %s", pdf_path)
    return pdf_path
