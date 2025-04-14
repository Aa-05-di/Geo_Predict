from bs4 import BeautifulSoup
import os
import pandas as pd

folder = "data"
captions = []

for file in os.listdir(folder):
    if file.endswith(".html"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            html_doc = f.read()

        soup = BeautifulSoup(html_doc, 'html.parser')

        # Extract the image alt text
        img_tag = soup.find('img', alt=True)
        if img_tag:
            alt_text = img_tag['alt'].strip()
            captions.append(alt_text)

# Create a DataFrame
df = pd.DataFrame(captions, columns=["Image Caption"])

# Write to CSV
df.to_csv("image_captions.csv", index=False, encoding="utf-8")
