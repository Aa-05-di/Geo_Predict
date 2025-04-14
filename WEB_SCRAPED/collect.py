from bs4 import BeautifulSoup
import os
import pandas as pd

folder = "./WEB_SCRAPED/data"
news_data = []

for file in os.listdir(folder):
    if file.endswith(".html"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            html_doc = f.read()

        soup = BeautifulSoup(html_doc, 'html.parser')

        headline = soup.find('h2', {'data-testid': 'card-headline'})
        description = soup.find('p', {'data-testid': 'card-description'})
        time = soup.find('span', {'data-testid': 'card-metadata-lastupdated'})
        tag = soup.find('span', {'data-testid': 'card-metadata-tag'})

        if headline and description and time and tag:
            news_data.append({
                'Headline': headline.text.strip(),
                'Description': description.text.strip(),
                'Time': time.text.strip(),
                'Category': tag.text.strip()
            })

# Create a DataFrame
df = pd.DataFrame(news_data)

# Write to CSV
df.to_csv("news_data.csv", index=False, encoding="utf-8")
