# scra.py
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def scrape_bbc_headlines(base_url, sections, output_folder):
    driver = webdriver.Safari()
    file_counter = 0
    for section in sections:
        url = f"{base_url}{section}/"
        driver.get(url)
        time.sleep(3)
        elem = driver.find_elements(By.CSS_SELECTOR, '[data-testid="dundee-article"]')
        print(f"{len(elem)} items found in {section}\n")
        for elems in elem:
            d = elems.get_attribute("outerHTML")
            with open(f"{output_folder}/article_{file_counter}_{section}.html", "w", encoding="utf-8") as f:
                f.write(d)
            file_counter += 1
            time.sleep(2)
    driver.close()

if __name__ == "__main__":
    base_url = "https://www.bbc.com/"
    sections = ['business', 'news']
    output_folder = "./data"
    scrape_bbc_headlines(base_url, sections, output_folder)