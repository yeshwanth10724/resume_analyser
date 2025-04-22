import joblib
from utils import preprocess_text
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def train_model():
    data = {
        'resume_text': [
            "Experienced Python developer with Flask skills.",
            "Java developer with extensive experience in Spring Boot.",
            "Data scientist with knowledge of machine learning and Python.",
            "Graphic designer with experience in Photoshop and Illustrator.",
            "Frontend developer with React and JavaScript skills."
        ],
        'job_match': [1, 0, 1, 0, 1] 
    }

    
    processed_resumes = [preprocess_text(resume) for resume in data['resume_text']]
    

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_resumes)
    y = data['job_match']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    joblib.dump(model, 'resume_classifier.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    print("Model training complete. Model saved to 'resume_classifier.pkl'.")

def predict_resume_match(resume_text):
    model = joblib.load('resume_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    preprocessed_text = preprocess_text(resume_text)
    resume_features = vectorizer.transform([preprocessed_text])
    prediction = model.predict(resume_features)
    return "Match" if prediction[0] == 1 else "No Match"
if __name__ == "__main__":
    sample_resume = "A Python developer with experience in Flask and Django."
    result = predict_resume_match(sample_resume)
    print(f"Prediction result for the sample resume: {result}")
