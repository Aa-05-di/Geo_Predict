# collect.py
from bs4 import BeautifulSoup
import os
import pandas as pd

def process_html_files(folder):
    news_data = []
    for file in os.listdir(folder):
        if file.endswith(".html"):
            try:
                with open(os.path.join(folder, file), "r", encoding="utf-8", errors="ignore") as f:
                    html_doc = f.read()
                soup = BeautifulSoup(html_doc, 'html.parser')
                headline = soup.find('h2', {'data-testid': 'card-headline'}) or \
                           soup.find('h1') or \
                           soup.find('h2')
                description = soup.find('p', {'data-testid': 'card-description'}) or \
                              soup.find('p', class_='description') or \
                              soup.find('p')
                time_element = soup.find('span', {'data-testid': 'card-metadata-lastupdated'}) or \
                               soup.find('time') or \
                               soup.find('span', class_='time')
                tag = soup.find('span', {'data-testid': 'card-metadata-tag'}) or \
                      soup.find('span', class_='tag') or \
                      soup.find('a', class_='category')
                news_data.append({
                    'Headline': headline.text.strip() if headline else "N/A",
                    'Description': description.text.strip() if description else "N/A",
                    'Time': time_element.text.strip() if time_element else "N/A",
                    'Category': tag.text.strip() if tag else "N/A",
                    'SourceFile': file
                })
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    return pd.DataFrame(news_data)

def save_to_csv(df, output_file):
    try:
        df.to_csv(output_file, index=False, encoding="utf-8")
        print(f"Successfully saved {len(df)} records to {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")

if __name__ == "__main__":
    folder = "./data"
    output_file = "news_data.csv"
    df = process_html_files(folder)
    save_to_csv(df, output_file)
    print("\nSummary:")
    print(f"Total files processed: {len(os.listdir(folder))}")
    print(f"Successful records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSample data:")
    print(df.head())