import os
import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin
import concurrent.futures
from tqdm import tqdm
from googletrans import Translator
from transformers import pipeline
import datetime
import hashlib
import signal
import sys

class IELTSConfig:
    def __init__(self):
        self.data_file = "ielts_dataset.json"
        self.cache_file = "ielts_cache.json"
        self.config_file = "config.json"  # Added config file
        self.categories = {
            "articles": ["article", "blog", "post"],
            "faq": ["faq", "questions", "help"],
            "guides": ["guide", "tutorial", "how-to"],
            "news": ["news", "updates"]
        }
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        self.running = True  # For graceful shutdown
        self.load_sources()
        # self.load_cache()
        # self.setup_models()

    def load_sources(self):
        """Load sources from external config file"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.sources = json.load(f)
        except FileNotFoundError:
            self.sources = {"persian": [], "international": []}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.sources, f, ensure_ascii=False, indent=4)

    # Rest of the class methods remain same until save_cache
    def save_cache(self):
        """Save cache and database with interruption handling"""
        if not self.running:
            print("Saving progress before shutdown...")
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, ensure_ascii=False, indent=4)

    # Rest of the class methods...

config = IELTSConfig()

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nReceived interrupt signal. Saving progress...")
    config.running = False
    config.save_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def clean_text(text):
    """Clean text content and remove sensitive information"""
    if not text:
        return ""
    # Remove phone numbers (various formats)
    text = re.sub(r'(\+\d{1,3}[-.\s]?)?(\d{2,4}[-.\s]?){2,3}\d{2,4}', '', text)
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove postal addresses (basic pattern)
    text = re.sub(r'\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|آدرس)\b', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)
    text = re.sub(r'\b(?:https?|www|ftp)\S+', '', text)
    return text.strip()

# Rest of the functions until main...

def main():
    print("\nStarting IELTS data collection process")
    print(f"Existing database: {len(config.database.get('data', []))} Q&A pairs")
    print(f"Sources: {len(config.sources['persian'])} Persian, {len(config.sources['international'])} International")

    all_new_data = []

    # Process Persian sources
    print("\nProcessing Persian sources...")
    for url in config.sources["persian"]:
        if not config.running:
            break
        try:
            response = requests.get(url, headers=config.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                    if a['href'] and not a['href'].startswith('#')]

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_url, link, "persian") for link in links[:20]]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    if not config.running:
                        executor.shutdown(wait=False)
                        break
                    all_new_data.extend(future.result())
        except Exception as e:
            print(f"Error processing source {url}: {str(e)}")

    # Process International sources
    print("\nProcessing International sources...")
    for url in config.sources["international"]:
        if not config.running:
            break
        try:
            response = requests.get(url, headers=config.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                    if a['href'] and '/article/' in a['href'].lower()]

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_url, link, "international") for link in links[:10]]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    if not config.running:
                        executor.shutdown(wait=False)
                        break
                    all_new_data.extend(future.result())
        except Exception as e:
            print(f"Error processing source {url}: {str(e)}")

    # Update database
    if all_new_data:
        update_database(all_new_data)
    config.save_cache()

    print(f"\nUpdate completed:")
    print(f"- New Q&A pairs added: {len(all_new_data)}")
    print(f"- Total Q&A in database: {len(config.database['data'])}")
    print(f"- Categories: {', '.join(config.database['metadata']['categories'])}")
    print(f"\nDatabase file: {config.data_file}")
    print(f"Cache file: {config.cache_file}")

if __name__ == "__main__":
    main()