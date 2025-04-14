from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Safari()
item = "article"
file = 0

# Base URL for world section
base_url = "https://www.bbc.com/"

# Array of sections to loop through
sections = ['business', 'news']

# Loop through each section
for section in sections:
    # Construct the URL for the section
    url = f"{base_url}{section}/"
    driver.get(url)
    time.sleep(3)  # Wait for the page to load
    
    # Find all the article links
    elem = driver.find_elements(By.CSS_SELECTOR, ".sc-c6f6255e-0.eGcloy")  # Adjust the selector if needed
    print(f"{len(elem)} items found in {section}\n")
    
    for elems in elem:
        # print(elems.text)
        # print("\n")
        d = elems.get_attribute("outerHTML")  # Get the full HTML of the article link
        print(d)
        with open(f"./data/{item}_{file}_{section}.html", "w", encoding="utf-8") as f:
            f.write(d)
            file += 1

    time.sleep(2)

driver.close()
