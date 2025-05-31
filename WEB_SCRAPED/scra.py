from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time

def scrape_bbc_headlines(base_url, sections, output_folder):
    driver = webdriver.Safari()
    os.makedirs(output_folder, exist_ok=True)
    file_counter = 0

    for section in sections:
        url = f"{base_url}{section}/"
        driver.get(url)
        time.sleep(3)

        cards = driver.find_elements(By.CSS_SELECTOR, '[data-testid="dundee-card"]')
        print(f"{len(cards)} articles found in {section}")

        for card in cards:
            try:
                headline_wrapper = card.find_element(By.CSS_SELECTOR, '[data-testid="card-text-wrapper"]')
                headline_divs = headline_wrapper.find_elements(By.CSS_SELECTOR, 'div')
                headline_text = ""
                for div in headline_divs:
                    text = div.text.strip()
                    if text:
                        headline_text = text
                        break  # Assume first non-empty div is the headline

                if not headline_text:
                    continue  # skip if no headline found

                # Save to file
                with open(f"{output_folder}/article_{file_counter}_{section}.txt", "w", encoding="utf-8") as f:
                    f.write(f"Headline: {headline_text}\n")

                file_counter += 1
                time.sleep(1)

            except Exception as e:
                print(f"Error parsing card: {e}")
                continue

    driver.quit()

if __name__ == "__main__":
    base_url = "https://www.bbc.com/"
    sections = ['news', 'business']
    output_folder = "./data"
    scrape_bbc_headlines(base_url, sections, output_folder)
