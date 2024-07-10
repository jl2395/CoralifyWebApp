from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained MobileNet model
try:
    model = load_model('my_mobilenet_model.h5')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Load the CSV file
try:
    coral_data = pd.read_csv('coral_data.csv')
    logging.info("CSV file loaded successfully")
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")
    coral_data = None

# Define class names
if coral_data is not None:
    class_names = coral_data['abbrev'].tolist()

# Serve the HTML form
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MobileNet Web App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #333;
        }
        .container {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 600px; /* Increased width for better table display */
        }
        h1 {
            margin-bottom: 20px;
        }
        input[type="file"] {
            display: block;
            margin: 0 auto 20px;
        }
        button {
            padding: 10px 15px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .progress-bar {
            background-color: #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        .progress-bar-inner {
            height: 24px;
            border-radius: 8px;
            background-color: #007bff;
            width: 0;
            text-align: center;
            color: #fff;
            line-height: 24px;
            transition: width 0.5s;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload an Image</h1>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="file" name="file" accept="image/*" required>
            <button type="submit">Upload</button>
        </form>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async function(event) {
            event.preventDefault();

            const fileInput = document.getElementById('file');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                let resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '';

                if (result.error) {
                    resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
                } else {
                    resultDiv.innerHTML = `
                        <h2>Prediction Result</h2>
                        <p><strong>Predicted Class:</strong> ${result['Predicted Class']}</p>
                        <p><strong>Probability:</strong> ${result['Probability'].toFixed(2)}%</p>

                        <h3>Prediction Probabilities</h3>
                        <table>
                            <tr><th>Class</th><th>Probability</th></tr>
                            ${result['Prediction Probabilities'].map(p => `
                                <tr>
                                    <td>${p.class}</td>
                                    <td>${(p.probability * 100).toFixed(2)}%</td>
                                </tr>
                            `).join('')}
                        </table>

                        <h3>Coral Description</h3>
                        <table>
                            <tr><th>Attribute</th><th>Value</th></tr>
                            <tr><td>Class Index</td><td>${result['Coral Description']['Class Index']}</td></tr>
                            <tr><td>Abbreviation</td><td>${result['Coral Description']['Abbreviation']}</td></tr>
                            <tr><td>Coral Name</td><td>${result['Coral Description']['Coral Name']}</td></tr>
                            <tr><td>Common Name</td><td>${result['Coral Description']['Common Name']}</td></tr>
                            <tr><td>Geographic Information</td><td>${result['Coral Description']['Geographic Information']}</td></tr>
                            <tr><td>Kingdom</td><td>${result['Coral Description']['Kingdom']}</td></tr>
                            <tr><td>Colour</td><td>${result['Coral Description']['Colour']}</td></tr>
                            <tr><td>Habitat</td><td>${result['Coral Description']['Habitat']}</td></tr>
                        </table>
                    `;
                }
            } catch (error) {
                console.error('Error:', error);
                let resultDiv = document.getElementById('result');
                resultDiv.innerHTML = `<p>Error: ${error.message}</p>`;
            }
        };
    </script>
</body>
</html>
'''


# Preprocess the image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Rescale the image
        logging.info(f'Preprocessed image array: {img_array}')
        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return None


# Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded image
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        temp_path = os.path.join(upload_folder, file.filename)
        file.save(temp_path)

        # Preprocess the image
        img_array = preprocess_image(temp_path)
        if img_array is None:
            return jsonify({'error': 'Error processing image'}), 500

        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        predictions = model.predict(img_array)[0]
        max_index = np.argmax(predictions)
        max_prob = predictions[max_index]

        # Get prediction probabilities
        prediction_probabilities = []
        for i, prob in enumerate(predictions):
            prediction_probabilities.append({
                'class': class_names[i],
                'probability': float(prob)
            })

        # Get coral description from the CSV file
        if coral_data is None:
            return jsonify({'error': 'CSV data not loaded'}), 500

        coral_info = coral_data.iloc[max_index]
        coral_description = {
            'Class Index': int(coral_info['index']),
            'Abbreviation': coral_info['abbrev'],
            'Coral Name': coral_info['coral_name'],
            'Common Name': coral_info['common name'],
            'Geographic Information': coral_info['geographic information'],
            'Kingdom': coral_info['kingdom'],
            'Colour': coral_info['colour'],
            'Habitat': coral_info['habitat']
        }

        # Rename the file with the predicted class
        new_filename = f"{class_names[max_index]}.jpg"
        new_path = os.path.join(upload_folder, new_filename)
        os.rename(temp_path, new_path)

        result = {
            'Predicted Class': class_names[max_index],
            'Probability': float(max_prob) * 100,
            'Prediction Probabilities': prediction_probabilities,
            'Coral Description': coral_description
        }

        logging.info(f'Predictions: {predictions}')
        logging.info(f'Result: {result}')

        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
