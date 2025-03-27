import os
from PIL import Image
import pytesseract
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel, CoherenceModel
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for the progress bar
import spacy.cli
spacy.cli.download("en_core_web_sm")
import spacy
nlp = spacy.load("en_core_web_sm")

# Set the path to the Tesseract executable (if not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    try:
        # Open the image
        image = Image.open(image_path)
        # Use pytesseract to extract text
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""

# Function to analyze text from a folder of images
def analyze_text_from_images(folder_path, max_images=100):
    all_text = ""
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))]
    total_images = min(len(image_files), max_images)  # Process only max_images

    # Process images with a loading bar
    for filename in tqdm(image_files[:total_images], desc="Processing images", unit="image"):
        image_path = os.path.join(folder_path, filename)
        text = extract_text_from_image(image_path)
        all_text += text + " "  # Combine text from all images

    print(f"Processed {total_images} images.")
    return all_text

# Preprocess the text data
def preprocess_text(text):
    text = re.sub('\s+', ' ', text)  # Remove extra spaces
    text = re.sub('\S*@\S*\s?', '', text)  # Remove emails
    text = re.sub('\'', '', text)  # Remove apostrophes
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabet characters
    text = text.lower()  # Convert to lowercase
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Lemmatization using spaCy
def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

# Main execution
if __name__ == "__main__":
    folder_path = r"working_dataset"
    combined_text = analyze_text_from_images(folder_path, max_images=10)  # Process only 10 images

    # Download NLTK stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    # Load spaCy model
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Preprocess the combined text
    tokens = preprocess_text(combined_text)
    lemmas = lemmatize(tokens)

    # Create dictionary and corpus
    id2word = corpora.Dictionary([lemmas])
    corpus = [id2word.doc2bow(lemmas)]

    # Build LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=20,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)

    # Print the topics
    topics = lda_model.print_topics(num_words=10)
    for idx, topic in topics:
        print(f'Topic: {idx}\nWords: {topic}\n')

    # Compute coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=[lemmas], dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
