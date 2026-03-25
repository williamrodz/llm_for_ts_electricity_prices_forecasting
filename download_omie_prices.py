import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import time

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

RETRY_COUNT = 2

OMIE_PAGES = {
    "Spain": "https://www.omie.es/en/file-access-list?parents=/Day-ahead%20Market/1.%20Prices&dir=%20Day-ahead%20market%20hourly%20prices%20in%20Spain&realdir=marginalpdbc",
    "Portugal": "https://www.omie.es/en/file-access-list?parents=/Day-ahead%20Market/1.%20Prices&dir=%20Day-ahead%20market%20hourly%20prices%20in%20Portugal&realdir=marginalpdbp"
}

# -----------------------------
# Helper functions
# -----------------------------
def fetch_download_links(page_url):
    """Scrape the OMIE directory page and extract the full download URLs."""
    resp = requests.get(page_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if href and "file-download" in href and href.endswith(".1"):
            # Convert relative href to full URL
            full_url = "https://www.omie.es" + href
            links.append(full_url)
    return links

def download_file(url):
    """Download a file with retries."""
    for attempt in range(1, RETRY_COUNT + 2):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            if attempt <= RETRY_COUNT:
                time.sleep(5)
            else:
                print(f"Failed to download {url} after {RETRY_COUNT+1} attempts.")
                return None

# -----------------------------
# Main function
# -----------------------------
def main():
    for market, page_url in OMIE_PAGES.items():
        print(f"Processing {market} files...")
        download_links = fetch_download_links(page_url)

        for link in tqdm(download_links, desc=f"Downloading {market}"):
            filename = link.split("filename=")[-1]
            save_path = RAW_DIR / filename
            if save_path.exists():
                continue  # Skip already downloaded

            content = download_file(link)
            if content:
                save_path.write_bytes(content)

    print(f"All downloads complete. Files saved in {RAW_DIR}")

if __name__ == "__main__":
    main()