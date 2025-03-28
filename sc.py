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
        self.config_file = "sources_config.json"
        self.categories = {
            "articles": ["article", "blog", "post"],
            "faq": ["faq", "questions", "help"],
            "guides": ["guide", "tutorial", "how-to"],
            "news": ["news", "updates"]
        }

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,fa;q=0.8"
        }

        self.load_sources()
        self.load_cache()
        self.setup_models()
        self.should_stop = False

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
        """Save cache and database with progress"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, ensure_ascii=False, indent=4)

    # ... (other existing methods) ...

def clean_text(text):
    """Clean text content and remove sensitive information"""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)
    text = re.sub(r'\b(?:https?|www|ftp)\S+', '', text)
    # Remove phone numbers
    text = re.sub(r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove postal addresses (basic pattern)
    text = re.sub(r'\b\d{1,5}\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', '', text)
    return text.strip()

# ... (extract_content, generate_qa_pairs, is_question remain same) ...

def process_url(url, source_type, config):
    """Process a single URL with pause checking"""
    if config.should_stop:
        return []
    content, context_sentences, category = extract_content(url)
    if not content:
        return []

    qa_pairs = generate_qa_pairs(content, context_sentences, source_type)
    config.mark_url_scraped(url, category)
    return [{
        **pair,
        "source_url": url,
        "category": category,
        "timestamp": datetime.datetime.now().isoformat()
    } for pair in qa_pairs]

def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful stopping"""
    print("\nStopping process... Saving progress...")
    config.should_stop = True

config = IELTSConfig()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\nStarting IELTS data collection process")
    print(f"Existing database: {len(config.database.get('data', []))} Q&A pairs")
    print(f"Sources: {len(config.sources['persian'])} Persian, {len(config.sources['international'])} International")

    all_new_data = []
    processed_urls = 0

    # Process Persian sources
    print("\nProcessing Persian sources...")
    for url in config.sources["persian"]:
        if config.should_stop:
            break
        try:
            response = requests.get(url, headers=config.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                    if a['href'] and not a['href'].startswith('#')]

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_url, link, "persian", config) for link in links[:20]]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    if config.should_stop:
                        break
                    all_new_data.extend(future.result())
                    processed_urls += 1
                    if processed_urls % 10 == 0:  # Save every 10 URLs
                        update_database(all_new_data)
                        config.save_cache()
                        print(f"Progress saved: {len(all_new_data)} new Q&A pairs")

        except Exception as e:
            print(f"Error processing source {url}: {str(e)}")

    # Process International sources
    print("\nProcessing International sources...")
    for url in config.sources["international"]:
        if config.should_stop:
            break
        try:
            response = requests.get(url, headers=config.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                    if a['href'] and '/article/' in a['href'].lower()]

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_url, link, "international", config) for link in links[:10]]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    if config.should_stop:
                        break
                    all_new_data.extend(future.result())
                    processed_urls += 1
                    if processed_urls % 10 == 0:
                        update_database(all_new_data)
                        config.save_cache()
                        print(f"Progress saved: {len(all_new_data)} new Q&A pairs")

        except Exception as e:
            print(f"Error processing source {url}: {str(e)}")

    # Final update
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