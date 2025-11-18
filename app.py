from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import PyPDF2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load trained model and preprocessing objects
model_path = "models/voting_model.pkl"
scaler_path = "models/scaler.pkl"
pca_path = "models/pca.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(pca_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
else:
    model, scaler, pca = None, None, None

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
    "worst fractal dimension"
]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="Error: No file uploaded.", feature_names=feature_names)

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="Error: No file selected.", feature_names=feature_names)
        
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                extracted_values = extract_values_from_pdf(file_path)
                if len(extracted_values) != 30:
                    raise ValueError("Extracted values do not match expected 30 features.")
                
                input_data = np.array(extracted_values).reshape(1, -1)
                print("Extracted Features:", input_data)
                input_data = scaler.transform(input_data)
                input_data = pca.transform(input_data)
                
               

                probabilities = model.predict_proba(input_data)[0]
                prediction = "Benign" if probabilities[1] > probabilities[0] else "Malignant"
                confidence = max(probabilities) * 100  # Convert to percentage
                
            except Exception as e:
                prediction = f"Error: {str(e)}"
    
    return render_template("index.html", prediction=prediction, confidence=confidence, feature_names=feature_names)

def extract_values_from_pdf(file_path):
    values = []
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            numbers = [float(num) for num in text.split() if num.replace('.', '', 1).isdigit()]
            values.extend(numbers)
    return values[:30]

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
