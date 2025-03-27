import os
from PIL import Image
import pytesseract
from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm  # For the loading bar
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF
import matplotlib.pyplot as plt  # For displaying the word cloud

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



# Function to generate and display a word cloud using the wordcloud package
def generate_word_cloud(text, output_path="word_cloud.svg"):
    if not text.strip():  # Check if the text is empty
        print("No text available to generate a word cloud.")
        return

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Display the word cloud using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")  # Turn off axis

    # Save the word cloud as a high-resolution SVG file
    plt.savefig(output_path, format='svg', dpi=1200)
    print(f"Word cloud saved as {output_path}")

    # Optionally, display the word cloud
    plt.show()

# Main execution
if __name__ == "__main__":
    folder_path = r"working_dataset"
    combined_text = analyze_text_from_images(folder_path, max_images=6318)  # Process only 10 images
    generate_word_cloud(combined_text, output_path="results/text_analysis/wordcloudfirst10.svg")  # Generate and save word cloud as SVG
