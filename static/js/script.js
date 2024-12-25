document.getElementById('url-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Show loading state
    const urlInput = document.getElementById('url-input').value;
    const resultDiv = document.getElementById('result');
    const predictionText = document.getElementById('prediction');
    const messageText = document.getElementById('message');
    resultDiv.classList.remove('hidden');
    predictionText.textContent = 'Prediction: Loading...';
    messageText.textContent = 'Please wait, analyzing the URL...';

    // Call the backend API
    const response = await fetch('http://your-render-app-url/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: urlInput })
    });

    const result = await response.json();

    if (response.ok) {
        predictionText.textContent = `Prediction: ${result.prediction}`;
        messageText.textContent = `The URL is ${result.prediction === 1 ? 'Malicious' : 'Safe'}.`;
    } else {
        predictionText.textContent = 'Prediction: Error';
        messageText.textContent = result.error || 'An error occurred while processing the URL.';
    }
});
