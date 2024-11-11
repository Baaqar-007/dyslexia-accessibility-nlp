let selectedFile = null;

// Trigger the hidden file input to upload an image
function triggerUpload() {
    document.getElementById('upload').click();
}

// Store the selected file in the `selectedFile` variable
document.getElementById('upload').addEventListener('change', function (event) {
    selectedFile = event.target.files[0];
    document.getElementById('resultText').innerText = `Image uploaded: ${selectedFile.name}`;
});

// Show loader and hide result text when processing
function showLoader() {
    document.getElementById('loader').style.display = 'block';
    document.getElementById('resultBox').style.display = 'none';
}

// Hide loader and show result text when done
function hideLoader() {
    document.getElementById('loader').style.display = 'none';
    document.getElementById('resultBox').style.display = 'block';
}

// Send the image to the backend model when 'Find Dyslexia' is clicked
async function findDyslexia() {
    if (!selectedFile) {
        alert('Please upload an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    showLoader();

    // Send the image to your backend endpoint
    try {
        const response = await fetch('https://your-backend-url.com/detect-dyslexia', {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) throw new Error('Failed to analyze image');

        const result = await response.json();
        hideLoader();
        document.getElementById('resultText').innerText = `Result: ${result.message || 'No dyslexia detected.'}`;
    } catch (error) {
        console.error('Error:', error);
        hideLoader();
        document.getElementById('resultText').innerText = 'Error: Could not retrieve result.';
    }
}
