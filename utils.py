import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm")
def preprocess_text(text):
    doc = nlp(text.lower())  
    processed_text = ""
    for token in doc:
        if token.text not in STOP_WORDS and not token.is_punct:
            processed_text += token.lemma_ + " "
    
    return processed_text.strip()
