import pandas as pd
import requests
from time import sleep
from tqdm import tqdm  # For progress bar

# Read the CSV file with the correct delimiter
df = pd.read_csv('resources/test_for_domain_finder.csv', delimiter=';')

# DeepSeek API configuration
DEEPSEEK_API_KEY = 'my API'  # Replace with your actual API key
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'

# Function to query DeepSeek API
def query_deepseek(tool_name):
    try:
        # Prepare the API request
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": "deepseek-chat",  # Replace with the correct model name
            "messages": [
                {"role": "user", "content": f"What is the official website, company name, and country of origin for {tool_name}?"}
            ],
            "temperature": 0.7,  # Optional: Adjust based on API requirements
            "max_tokens": 150  # Optional: Adjust based on API requirements
        }

        # Make the API request
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse the API response
        data = response.json()
        print(f"API Response for {tool_name}: {data}")  # Debugging: Print the API response

        # Check if the response contains the expected structure
        if 'choices' in data and len(data['choices']) > 0:
            message = data['choices'][0].get('message', {}).get('content', '')
            if message:
                # Extract domain, company, and country from the message
                domain = None
                company = None
                country = None

                # Extract domain
                if "**Official Website**:" in message:
                    domain = message.split("**Official Website**:")[1].split("\n")[0].strip()
                    domain = domain.split("](")[1].split(")")[0].strip()  # Extract URL from markdown link

                # Extract company
                if "**Company Name**:" in message:
                    company = message.split("**Company Name**:")[1].split("\n")[0].strip()

                # Extract country
                if "**Country of Origin**:" in message:
                    country = message.split("**Country of Origin**:")[1].split("\n")[0].strip()

                print(f"Found data for {tool_name}: Domain={domain}, Company={company}, Country={country}")
                return domain, company, country
            else:
                print(f"No message content found for {tool_name}")
        else:
            print(f"No choices found in the API response for {tool_name}")
    except requests.exceptions.RequestException as e:
        print(f"Error querying DeepSeek API for {tool_name}: {e}")
    except Exception as e:
        print(f"Unexpected error for {tool_name}: {e}")
    return None, None, None

# Add new columns to the DataFrame
df['Web Domain'] = None
df['Company'] = None
df['Country'] = None

# Apply the function to each row in the DataFrame with a progress bar
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing tools"):
    tool_name = row['Tool Name']
    domain, company, country = query_deepseek(tool_name)
    df.at[index, 'Web Domain'] = domain
    df.at[index, 'Company'] = company
    df.at[index, 'Country'] = country
    sleep(1)  # Add a delay to avoid hitting API rate limits

# Save the updated DataFrame to a new CSV file
df.to_csv('ai_tools_with_domains_and_details.csv', index=False, sep=';')  # Use the same delimiter for output

print("CSV file with domains, company names, and countries has been created.")
