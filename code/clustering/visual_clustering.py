import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')  # Or 'Agg' if running in a script without GUI

# Define image directory
IMAGE_DIR = r"C:\Users\juliu\Documents\Coding Projects\Center for Digital Participation\toolify.ai_image_scraping\ai_website_images\normalized_working_dataset"

# Load pre-trained VGG16 model (without top layers)
model = VGG16(weights='imagenet', include_top=False)


def extract_features(image_path):
    """Extracts features from an image using VGG16."""
    img = load_img(image_path, target_size=(360, 640))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


# Load images and extract features
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('png', 'jpg', 'jpeg'))]
features = []

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    features.append(extract_features(img_path))

# Convert feature list to NumPy array
features = np.array(features)

# Check if features were extracted successfully
if features.shape[0] == 0:
    raise ValueError("No features extracted. Check if the image directory contains valid images.")

# Reduce dimensionality using PCA
pca = PCA(n_components=50)
reduced_features = pca.fit_transform(features)

# Cluster using K-Means
num_clusters = 20  # Change based on your needs
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(reduced_features)
labels = kmeans.labels_

# Visualize clusters using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_features = tsne.fit_transform(reduced_features)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=labels, palette='viridis')
plt.title("Image Clustering Visualization")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.savefig("normalized_working_dataset_200_sample.svg", format='svg', dpi=300)

# Group images by cluster labels
clustered_images = {i: [] for i in range(num_clusters)}
for img_file, label in zip(image_files, labels):
    clustered_images[label].append(img_file)


# Function to create a collage for a cluster
def create_collage(cluster_images, cluster_label, images_per_row=5):
    num_images = len(cluster_images)
    num_rows = (num_images // images_per_row) + (1 if num_images % images_per_row != 0 else 0)

    # Adjust figure size to fit the images tightly
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 3 * num_rows))
    fig.suptitle(f"Cluster {cluster_label}", fontsize=16)

    # Flatten axes if there's only one row
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for i, img_file in enumerate(cluster_images):
        img_path = os.path.join(IMAGE_DIR, img_file)
        img = load_img(img_path, target_size=(360, 640))  # Resize images for collage
        ax = axes[i // images_per_row, i % images_per_row]
        ax.imshow(img)
        ax.axis('off')  # Turn off axis labels and ticks

    # Hide empty subplots
    for i in range(num_images, num_rows * images_per_row):
        ax = axes[i // images_per_row, i % images_per_row]
        ax.axis('off')

    # Adjust layout to minimize gaps
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=0.9)
    plt.tight_layout()
    plt.savefig(f"cluster_{cluster_label}_collage.png", format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


# Create collages for each cluster
for cluster_label, cluster_images in clustered_images.items():
    create_collage(cluster_images, cluster_label)