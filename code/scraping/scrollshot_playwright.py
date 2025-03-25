import pandas as pd
from tqdm import tqdm
from playwright.sync_api import sync_playwright
import os
import time


def take_scrollshot(url, save_folder):
    """
    Takes a full-page screenshot of the given URL and saves it to the specified folder.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Go to the URL and wait for the network to be idle
            page.goto(url)
            page.wait_for_load_state('networkidle')

            # Introduce a delay to ensure all content is loaded
            time.sleep(5)  # Adjust the delay as needed

            # Ensure the save folder exists
            os.makedirs(save_folder, exist_ok=True)

            # Extract the domain name for the filename
            domain = url.replace("https://", "").replace("http://", "").replace("/", "_")

            # Remove trailing underscores
            domain = domain.rstrip("_")

            # Construct the screenshot path
            screenshot_path = os.path.join(save_folder, f"{domain}.png")

            # Take the full-page screenshot
            page.screenshot(path=screenshot_path, full_page=True)

            # Close the browser
            browser.close()

            print(f"Saved screenshot for {url} at {screenshot_path}")
    except Exception as e:
        print(f"Failed to take screenshot for {url}: {e}")


if __name__ == "__main__":
    # Load the CSV file
    csv_path = 'resources/ai_tools_with_domain_from_duckduckgo.csv'
    try:
        df = pd.read_csv(csv_path, delimiter=';', encoding='latin-1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, delimiter=';', encoding='ISO-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, delimiter=';', encoding='utf-8')

    # Define the save folder for scrollshots
    save_folder = r"C:\Users\juliu\Documents\Coding Projects\Center for Digital Participation\toolify.ai_image_scraping\ai_website_images\new_scrollshot_first_20_2"

    # Define the starting index (e.g., 5858)
    start_index = 0

    # Process rows starting from the specified index
    for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df) - start_index, desc="Taking scrollshots"):
        tool_name = row['Tool Name']
        web_domain = row['Web Domain']

        if pd.notna(web_domain):  # Check if the Web Domain is not NaN
            print(f"Processing {tool_name} ({web_domain})...")
            take_scrollshot(web_domain, save_folder)
        else:
            print(f"Skipping {tool_name} (no Web Domain available).")

    print("Scrollshots completed for all remaining domains.")