import pandas as pd
from duckduckgo_search import DDGS
from time import sleep
from tqdm import tqdm  # For progress bar

# Read the CSV file with the correct delimiter and encoding
try:
    df = pd.read_csv('resources/cleaned_combined_region_data.csv', delimiter=';', encoding='latin-1')  # Try 'latin-1' or 'ISO-8859-1'
except UnicodeDecodeError:
    try:
        df = pd.read_csv('resources/cleaned_combined_region_data.csv', delimiter=';', encoding='ISO-8859-1')  # Try 'ISO-8859-1'
    except UnicodeDecodeError:
        df = pd.read_csv('resources/cleaned_combined_region_data.csv', delimiter=';', encoding='utf-8')  # Fallback to 'utf-8'

# Function to search DuckDuckGo and get the first result URL
def get_first_result_url(tool_name, retries=3, delay=5):
    for attempt in range(retries):
        try:
            # Perform a DuckDuckGo search
            with DDGS() as ddgs:
                query = f"{tool_name} AI Tool website"  # Add "AI Tool website" to the query
                results = list(ddgs.text(query, max_results=1))
                print(f"Search results for {query}: {results}")  # Debugging: Print the search results
                if results:
                    first_result = results[0]
                    url = first_result.get('href', None)  # Use 'href' instead of 'link'
                    print(f"Found URL for {tool_name}: {url}")
                    return url
        except Exception as e:
            print(f"Error searching DuckDuckGo for {tool_name} (attempt {attempt + 1}): {e}")
            sleep(delay * (attempt + 1))  # Increase delay with each retry
    return None

# Add a new column for the Web Domain
df['Web Domain'] = None

# Backup interval (save every 200 rows)
backup_interval = 200

# Apply the function to each row in the DataFrame with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing tools"):
    tool_name = row['Tool Name']
    url = get_first_result_url(tool_name)
    df.at[index, 'Web Domain'] = url
    sleep(3)  # Add a longer delay to avoid hitting the ratelimit

    # Save a backup every 200 rows
    if (index + 1) % backup_interval == 0:
        backup_filename = f"ai_tools_with_domain_backup_{index + 1}.csv"
        df.to_csv(backup_filename, index=False, sep=';')
        print(f"Backup saved to {backup_filename}")

# Save the final updated DataFrame to a new CSV file
final_filename = 'ai_tools_with_domain_from_duckduckgo.csv'
df.to_csv(final_filename, index=False, sep=';')  # Use the same delimiter for output
print(f"Final CSV file saved as {final_filename}")

print("CSV file with domains has been created.")