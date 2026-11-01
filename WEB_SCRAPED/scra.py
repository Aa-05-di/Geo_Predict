import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions


def get_driver(headless: bool = True):
    """Initializes headless browser driver."""
    options = FirefoxOptions()
    if headless:
        options.add_argument("--headless")
    return webdriver.Firefox(options=options)


def scrape_bbc_headlines(
    base_url: str = "https://www.bbc.com/",
    sections: list = None,
    output_folder: str = "WEB_SCRAPED/data",
    headless: bool = True
):
    """
    Scrapes BBC headline cards from the specified sections and saves
    the card HTML snippets to the output directory for parsing by collect.py.
    """
    if sections is None:
        sections = ["business", "news"]

    out_path = Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    # Clean previous scrape files
    for old_file in out_path.glob("*.html"):
        try:
            old_file.unlink()
        except OSError:
            pass

    driver = get_driver(headless=headless)
    file_counter = 0

    try:
        for section in sections:
            url = f"{base_url.rstrip('/')}/{section}/"
            print(f"Scraping {url}...")
            driver.get(url)
            time.sleep(3)

            cards = driver.find_elements(By.CSS_SELECTOR, '[data-testid="dundee-card"]')
            print(f"  Found {len(cards)} articles in /{section}/")

            for card in cards:
                try:
                    html = card.get_attribute("outerHTML")
                    if html:
                        article_file = out_path / f"article_{file_counter}_{section}.html"
                        with open(article_file, "w", encoding="utf-8") as f:
                            f.write(html)
                        file_counter += 1
                except Exception:
                    continue

        print(f"✅ Scraped {file_counter} article cards into '{output_folder}'.")
    finally:
        driver.quit()

    return file_counter


if __name__ == "__main__":
    scrape_bbc_headlines()
