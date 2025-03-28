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

class ConfigLoader:
    @staticmethod
    def load_config(config_file="config.json"):
        """Load configuration from external JSON file"""
        default_config = {
            "data_file": "ielts_dataset.json",
            "cache_file": "ielts_cache.json",
            "categories": {
                "articles": ["article", "blog", "post"],
                "faq": ["faq", "questions", "help"],
                "guides": ["guide", "tutorial", "how-to"],
                "news": ["news", "updates"]
            },
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9,fa;q=0.8"
            },
            "sources": {
                "persian": [
                    "https://irsafam.org",
                    "https://gosafir.com/",
                    "https://www.ielts4migration.com/",
                    "https://www.proshotportal.com/",
                    "https://armanienglish.com/",
                    "https://www.idp.com/iran/ielts/",
                    "https://afarinesh.org/"
                ],
                "international": []
            }
        }
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using default configuration")
            return default_config
        except json.JSONDecodeError:
            print(f"Invalid config file {config_file}, using default configuration")
            return default_config

class IELTSConfig:
    def __init__(self):
        config_data = ConfigLoader.load_config()
        
        self.data_file = config_data.get("data_file", "ielts_dataset.json")
        self.cache_file = config_data.get("cache_file", "ielts_cache.json")
        self.categories = config_data.get("categories", {})
        self.headers = config_data.get("headers", {})
        self.sources = config_data.get("sources", {})
        
        self.load_cache()
        self.setup_models()
        self.setup_interrupt_handler()
        
        # Track progress for resuming
        self.current_source = None
        self.current_urls = []
        self.processed_urls = set()
    
    def setup_interrupt_handler(self):
        """Setup handler for graceful interruption"""
        def signal_handler(sig, frame):
            print("\nInterrupt received, saving progress...")
            self.save_cache()
            print(f"Progress saved. Data file: {self.data_file}, Cache file: {self.cache_file}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def setup_models(self):
        """Initialize NLP models and translator"""
        try:
            self.translator = Translator(service_urls=['translate.googleapis.com'])
            test_trans = self.translator.translate("hello", src='en', dest='fa').text
            assert test_trans == "سلام"
        except:
            self.translator = None
        
        try:
            self.nlp_model = pipeline(
                "text-classification",
                model="HooshvareLab/bert-fa-base-uncased",
                device=-1
            )
        except:
            self.nlp_model = None
    
    def load_cache(self):
        """Load existing cache and database"""
        self.cache = {"scraped_urls": {}, "last_checked": {}}
        self.database = {"data": [], "metadata": {}}
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
        
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.database = json.load(f)
    
    def save_cache(self):
        """Save cache and database"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)
        
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, ensure_ascii=False, indent=4)
    
    def get_url_hash(self, url):
        """Generate unique hash for URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def is_url_scraped(self, url):
        """Check if URL has been processed before"""
        url_hash = self.get_url_hash(url)
        return url_hash in self.cache["scraped_urls"]
    
    def mark_url_scraped(self, url, category):
        """Mark URL as processed"""
        url_hash = self.get_url_hash(url)
        self.cache["scraped_urls"][url_hash] = {
            "url": url,
            "category": category,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.processed_urls.add(url)
    
    def detect_category(self, url, content):
        """Detect content category"""
        for category, keywords in self.categories.items():
            if any(kw in url.lower() for kw in keywords):
                return category
            if any(kw in content.lower() for kw in keywords):
                return category
        return "general"

config = IELTSConfig()

def clean_text(text):
    """Clean text content and remove contact information"""
    if not text:
        return ""
    
    # Remove contact information
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL_REMOVED]', text)  # Emails
    text = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE_REMOVED]', text)  # Phone numbers
    text = re.sub(r'\b\d{1,5}\s+\w+\s+\w+,\s*\w+,\s*\w+\s+\d{5}(?:-\d{4})?\b', '[ADDRESS_REMOVED]', text)  # Addresses
    
    # General cleaning
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)
    text = re.sub(r'\b(?:https?|www|ftp)\S+', '', text)
    return text.strip()

def extract_content(url):
    """Extract content from URL with caching"""
    if config.is_url_scraped(url):
        return None, None, None
    
    try:
        response = requests.get(url, headers=config.headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for tag in ['script', 'style', 'iframe', 'nav', 'footer', 'header']:
            for element in soup(tag):
                element.decompose()
        
        # Extract main content
        paragraphs = []
        for tag in ['article', 'main', 'section', 'div', 'p']:
            for element in soup.find_all(tag):
                text = clean_text(element.get_text())
                if text and len(text.split()) > 5:
                    paragraphs.append(text)
        
        full_content = ' '.join(paragraphs)
        category = config.detect_category(url, full_content)
        
        # Extract context (3-5 sentences around each question)
        context_sentences = re.split(r'(?<=[.!?])\s+', full_content)
        return full_content, context_sentences, category
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None, None, None

def generate_qa_pairs(content, context_sentences, source_type):
    """Generate QA pairs with context"""
    if not content or not context_sentences:
        return []
    
    qa_pairs = []
    sentences = [s.strip() for s in re.split(r'([.!?])', content) if s.strip()]
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) and 
                re.match(r'^[.!?]$', sentences[i+1]) else '') 
                for i in range(0, len(sentences), 2)]
    
    for i in range(len(sentences)-1):
        question = sentences[i]
        answer = sentences[i+1] if i+1 < len(sentences) else ""
        
        if is_question(question) and len(answer.split()) > 3:
            # Find relevant context (3 sentences before and after)
            context_start = max(0, i-3)
            context_end = min(len(context_sentences), i+4)
            context = ' '.join(context_sentences[context_start:context_end])
            
            if source_type == "international" and config.translator:
                try:
                    question = config.translator.translate(question, src='en', dest='fa').text
                    answer = config.translator.translate(answer, src='en', dest='fa').text
                    context = config.translator.translate(context, src='en', dest='fa').text
                except:
                    pass
            
            qa_pairs.append({
                "context": clean_text(context),
                "question": clean_text(question),
                "answer": clean_text(answer),
                "source": source_type
            })
    
    return qa_pairs

def is_question(text):
    """Check if text is a question"""
    if len(text.split()) < 3:
        return False
    
    # Regex patterns
    question_patterns = [
        r"(آیا|چطور|چگونه|چرا|چه|کی|کجا|چقدر|آیا می‌دانید).*\?",
        r"(هزینه|زمان|نمره|منبع|ثبت‌نام|لیسنینگ|ریدینگ|رایتینگ|اسپیکینگ).*\??",
        r"\?$"
    ]
    
    pattern_check = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
    
    # NLP model check if available
    if config.nlp_model:
        try:
            result = config.nlp_model(text[:512])[0]
            return (result['label'] == 'LABEL_1' and result['score'] > 0.6) or pattern_check
        except:
            return pattern_check
    
    return pattern_check

def process_url(url, source_type):
    """Process a single URL"""
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

def update_database(new_data):
    """Update database with new data"""
    existing_hashes = {hashlib.md5((q['question'] + q['answer']).encode('utf-8')).hexdigest() 
                      for q in config.database.get("data", [])}
    
    for item in new_data:
        item_hash = hashlib.md5((item['question'] + item['answer']).encode('utf-8')).hexdigest()
        if item_hash not in existing_hashes:
            config.database["data"].append(item)
            existing_hashes.add(item_hash)
    
    # Update metadata
    config.database["metadata"] = {
        "last_updated": datetime.datetime.now().isoformat(),
        "total_qa": len(config.database["data"]),
        "categories": list(set(q["category"] for q in config.database["data"]))
    }

def main():
    print("\nStarting IELTS data collection process")
    print(f"Existing database: {len(config.database.get('data', []))} Q&A pairs")
    print(f"Sources: {len(config.sources['persian'])} Persian, {len(config.sources['international'])} International")
    
    all_new_data = []
    
    # Process Persian sources
    print("\nProcessing Persian sources...")
    for source_url in config.sources["persian"]:
        try:
            config.current_source = source_url
            print(f"Processing source: {source_url}")
            
            response = requests.get(source_url, headers=config.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [urljoin(source_url, a['href']) for a in soup.find_all('a', href=True) 
                    if a['href'] and not a['href'].startswith('#')]
            
            # Filter out already processed URLs
            links = [link for link in links if link not in config.processed_urls]
            config.current_urls = links
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_url, link, "persian") for link in links[:20]]  # Limit for testing
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    all_new_data.extend(future.result())
                    
                    # Save progress periodically
                    if len(all_new_data) % 10 == 0:
                        update_database(all_new_data)
                        all_new_data = []
                        config.save_cache()
        except Exception as e:
            print(f"Error processing source {source_url}: {str(e)}")
    
    # Process International sources
    print("\nProcessing International sources...")
    for source_url in config.sources["international"]:
        try:
            config.current_source = source_url
            print(f"Processing source: {source_url}")
            
            response = requests.get(source_url, headers=config.headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [urljoin(source_url, a['href']) for a in soup.find_all('a', href=True) 
                    if a['href'] and '/article/' in a['href'].lower()]
            
            # Filter out already processed URLs
            links = [link for link in links if link not in config.processed_urls]
            config.current_urls = links
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_url, link, "international") for link in links[:10]]  # Limit for testing
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    all_new_data.extend(future.result())
                    
                    # Save progress periodically
                    if len(all_new_data) % 10 == 0:
                        update_database(all_new_data)
                        all_new_data = []
                        config.save_cache()
        except Exception as e:
            print(f"Error processing source {source_url}: {str(e)}")
    
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