import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from colorsys import rgb_to_hsv
from collections import defaultdict
from tqdm import tqdm  # For the loading bar

# Function to resize image
def resize_image(image, size=(256, 256)):
    return image.resize(size, Image.ANTIALIAS)  # Use ANTIALIAS for better quality

# Function to filter out background colors (e.g., white)
def filter_background_colors(pixels, background_threshold=0.9):
    # Convert RGB to grayscale intensity
    intensity = pixels.mean(axis=1)
    # Filter out pixels with high intensity (e.g., white)
    return pixels[intensity < background_threshold * 255]

# Function to filter out low-saturation colors
def filter_low_saturation_colors(pixels, saturation_threshold=0.2):
    hsv_pixels = np.array([rgb_to_hsv(*pixel) for pixel in pixels / 255])
    saturation = hsv_pixels[:, 1]
    return pixels[saturation > saturation_threshold]

# Function to extract dominant theme colors using K-Means
from sklearn.cluster import MiniBatchKMeans  # Use MiniBatchKMeans

def get_dominant_theme_colors(image, num_colors=5, background_threshold=0.9, saturation_threshold=0.2):
    image = image.convert("RGB")
    image_array = np.array(image)
    pixels = image_array.reshape(-1, 3)

    # Filter out background and low-saturation colors
    pixels = filter_background_colors(pixels, background_threshold)
    pixels = filter_low_saturation_colors(pixels, saturation_threshold)

    if len(pixels) == 0:
        return np.array([]), np.array([])  # No colors left after filtering

    # Use MiniBatchKMeans instead of regular KMeans
    kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=42, batch_size=2048, n_init="auto")
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    cluster_sizes = np.bincount(kmeans.labels_)  # Size of each cluster
    return colors, cluster_sizes


# Function to analyze color themes in a folder
def analyze_color_themes(folder_path, num_colors=5, max_images=100):
    hue_counter = defaultdict(float)  # Use float for weighted counts

    # Get list of image files
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))]
    total_images = min(len(image_files), max_images)  # Process only max_images

    # Process images with a loading bar
    for filename in tqdm(image_files[:total_images], desc="Processing images", unit="image"):
        image_path = os.path.join(folder_path, filename)
        try:
            image = Image.open(image_path)
            image = resize_image(image)
            dominant_colors, cluster_sizes = get_dominant_theme_colors(image, num_colors)
            if len(dominant_colors) > 0:
                # Normalize cluster sizes to sum to 1
                cluster_weights = cluster_sizes / cluster_sizes.sum()
                for color, weight in zip(dominant_colors, cluster_weights):
                    # Convert RGB to HSV and extract hue
                    h, s, v = rgb_to_hsv(*(color / 255))
                    hue_counter[int(h * 360)] += weight  # Add weighted hue
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processed {total_images} images.")
    return hue_counter

# Function to display and save color themes as a histogram
def display_color_histogram(hue_counter, save_path="color_histogram.svg"):
    # Ensure we have all 360 hue values (0-359) represented
    hues = np.arange(360)  # Create an array of all possible hue values
    frequencies = np.array([hue_counter.get(h, 0) for h in hues])  # Fill missing hues with 0

    # Create a color gradient for the x-axis
    colors = [plt.cm.hsv(hue / 360) for hue in hues]

    # Plot the histogram
    plt.figure(figsize=(15, 6), dpi=300)  # High resolution
    plt.bar(hues, frequencies, color=colors, width=1)  # Each bar is exactly 1 degree wide
    plt.xlabel("Hue (0-360 degrees)")
    plt.ylabel("Weighted Prevalence")
    plt.title("Color Prevalence by Hue (Weighted by Dominance)")
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save as a high-resolution SVG file
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    plt.show()

# Main execution
if __name__ == "__main__":
    folder_path = r"path_to_working_dataset"  # Replace with your folder path
    hue_counter = analyze_color_themes(folder_path, max_images=6318)  # Process only 10 images
    display_color_histogram(hue_counter, save_path="results/color_analysis/first_10_test_hue_dominance.svg")
