from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import PyPDF2

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")
KEYWORDS = ["python", "flask", "react", "api", "mongodb", "nlp"]

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

@app.route('/upload-resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    text = extract_text(file).lower()
    matched = [kw for kw in KEYWORDS if kw in text]
    score = round(len(matched) / len(KEYWORDS) * 100, 2)
    suggestions = list(set(KEYWORDS) - set(matched))

    return jsonify({
        "score": score,
        "suggestions": suggestions
    })

if __name__ == '__main__':
    app.run(debug=True)
