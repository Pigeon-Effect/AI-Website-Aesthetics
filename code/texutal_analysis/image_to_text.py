import os
import csv
from PIL import Image
import pytesseract
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return image_path, text.replace(';', ',').strip()
    except Exception as e:
        return image_path, f"Error: {e}"


def process_images_to_csv(folder_path, output_csv_path, max_images=None):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif'))]

    if max_images:
        image_files = image_files[:max_images]

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(extract_text_from_image, image_files), total=len(image_files), desc="Processing images",
                 unit="image"))

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['path', 'text'])
        writer.writerows(results)

    print(f"\nSuccessfully processed {len(image_files)} images to {output_csv_path}")


if __name__ == "__main__":
    input_folder = r"working_dataset"
    output_csv = r"image_to_text_results.csv"
    process_images_to_csv(input_folder, output_csv, max_images=None)
