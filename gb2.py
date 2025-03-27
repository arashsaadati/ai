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

# تنظیمات پیشرفته
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8"
}

# منابع فارسی معتبر
persian_sources = [
    "https://irsafam.org",
    "https://zaban24.com/ielts",
    "https://ielts-rooz.ir",
    "https://www.ieltscity.com"
]

# منابع بین‌المللی
international_sources = [
    "https://www.ielts.org",
    "https://takeielts.britishcouncil.org"
]

# مترجم
translator = Translator(service_urls=['translate.googleapis.com'])

# مدل سبک‌تر برای تشخیص سوالات (DistilBERT)
try:
    question_detector = pipeline(
        "text-classification",
        model="distilbert-base-uncased",  # مدل سبک‌تر
        tokenizer="distilbert-base-uncased",
        device=-1  # استفاده از CPU
    )
    print("✅ مدل DistilBERT با موفقیت بارگذاری شد")
except Exception as e:
    print(f"⚠️ خطا در بارگذاری مدل: {str(e)}")
    question_detector = None

# الگوهای پیشرفته سوالات (بهینه‌شده)
question_patterns = [
    r"(آیلتس\s*چی[ست]|چیست\??)",
    r"(چگونه|چطور|روش)\s.*\?",
    r"(هزینه|قیمت|مبلغ)\s.*\?",
    r"(نمره|امتیاز)\s.*\?",
    r"(منابع|کتاب|تمرین)\s.*\?",
    r"(تفاوت|فرق)\s.*\?",
    r"(شرایط|نیازمندی)\s.*\?",
    r"(آمادگی|آموزش)\s.*\?",
    r"(مدت|زمان)\s.*\?",
    r"(چرا|علت|دلیل)\s.*\?",
    r"(آیا|ایا)\s.*\?",
    r"(کدام|چه|چگونه|چرا|کی|کجا)\s.*\?"
]

# کلیدواژه‌های مرتبط
ielts_keywords = [
    'آیلتس', 'ielts', 'ریدینگ', 'رایتینگ',
    'لیسنینگ', 'اسپیکینگ', 'نمره', 'مهاجرت',
    'تحصیل', 'آزمون', 'زبان انگلیسی'
]

def is_relevant(url, text):
    """بررسی ارتباط محتوا با آیلتس"""
    text = text.lower()
    url_check = any(kw in url.lower() for kw in ielts_keywords)
    content_check = any(kw in text[:300] for kw in ielts_keywords)
    return url_check or content_check

def clean_text(text):
    """پاکسازی متن"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\b(?:https?|www)\S+', '', text)
    return text.strip()

def extract_links(url):
    """استخراج لینک‌های مرتبط"""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            text = a.get_text().lower()
            if any(kw in href or kw in text for kw in ielts_keywords):
                full_url = urljoin(url, href)
                if not any(ext in full_url for ext in ['.pdf', '.jpg', '.png']):
                    links.add(full_url)
        return list(links)[:15]  # حداکثر 15 لینک
    except Exception as e:
        print(f"خطا در استخراج لینک از {url}: {str(e)}")
        return []

def scrape_page(url):
    """اسکرپ محتوای صفحه"""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # حذف عناصر غیرمفید
        for tag in ['script', 'style', 'iframe', 'nav', 'footer', 'header']:
            for element in soup(tag):
                element.decompose()
        
        # استخراج محتوای اصلی
        content = []
        for tag in ['p', 'div', 'article', 'section']:
            for element in soup.find_all(tag):
                text = clean_text(element.get_text())
                if len(text.split()) > 5 and is_relevant(url, text):
                    content.append(text)
        
        return ' '.join(content)
    except Exception as e:
        print(f"خطا در اسکرپ {url}: {str(e)}")
        return ""

def is_question(text):
    """تشخیص سوال با ترکیب الگوها و مدل"""
    if len(text.split()) < 3:
        return False
    
    has_question_mark = '؟' in text or '?' in text
    pattern_check = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
    
    if question_detector and len(text) > 10:
        try:
            result = question_detector(text[:512])
            return (result[0]['label'] == 'LABEL_1' and result[0]['score'] > 0.7) or pattern_check
        except:
            return pattern_check and has_question_mark
    
    return pattern_check and has_question_mark

def generate_qa(text, source_type):
    """تولید سوال-پاسخ"""
    qa_pairs = []
    sentences = [s.strip() for s in re.split(r'([؟?!\.])', text) if s.strip()]
    sentences = [s + sentences[i+1] if i+1 < len(sentences) and re.match(r'^[؟?!\.]$', sentences[i+1]) else s 
               for i, s in enumerate(sentences) if not re.match(r'^[؟?!\.]$', s)]
    
    for i in range(len(sentences)-1):
        current = sentences[i]
        next_sent = sentences[i+1] if i+1 < len(sentences) else ""
        
        if is_question(current) and len(next_sent.split()) > 3:
            if source_type == "international":
                current = translate_to_persian(current)
                next_sent = translate_to_persian(next_sent)
            
            qa_pairs.append({
                "question": current,
                "answer": next_sent
            })
    
    return qa_pairs

def process_source(url, source_type):
    """پردازش یک منبع"""
    links = extract_links(url)
    all_qa = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(scrape_page, link): link for link in links}
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(links), 
                         desc=f"پردازش {url.split('//')[1][:15]}..."):
            content = future.result()
            if content:
                all_qa.extend(generate_qa(content, source_type))
    
    return all_qa

def main():
    print("\n🔍 شروع جمع‌آوری داده‌های آیلتس...")
    
    dataset = {
        "metadata": {
            "version": "2.2",
            "model": "DistilBERT" if question_detector else "Regex",
            "created_at": datetime.datetime.now().isoformat(),
            "sources": persian_sources + international_sources
        },
        "data": []
    }
    
    # پردازش منابع فارسی
    print("\n📚 پردازش منابع فارسی...")
    for source in tqdm(persian_sources, desc="منابع فارسی"):
        dataset["data"].extend(process_source(source, "persian"))
    
    # پردازش منابع بین‌المللی
    print("\n🌍 پردازش منابع بین‌المللی...")
    for source in tqdm(international_sources, desc="منابع خارجی"):
        dataset["data"].extend(process_source(source, "international"))
    
    # حذف تکراری‌ها
    unique_qa = []
    seen = set()
    for item in dataset["data"]:
        key = (item["question"][:100], item["answer"][:100])  # جلوگیری از حافظه زیاد
        if key not in seen:
            seen.add(key)
            unique_qa.append(item)
    
    dataset["data"] = unique_qa
    dataset["metadata"]["total_qa"] = len(unique_qa)
    
    # ذخیره دیتاست
    output_file = "ielts_dataset_light.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ دیتاست با {len(unique_qa)} سوال-پاسخ در '{output_file}' ذخیره شد.")
    print(f"⚡ مدل استفاده شده: {dataset['metadata']['model']}")

if __name__ == "__main__":
    main()