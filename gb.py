import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin
import concurrent.futures
from tqdm import tqdm
from googletrans import Translator
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import datetime

# تنظیمات پیشرفته
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "fa-IR,fa;q=0.9"
}

persian_sources = [
    "https://irsafam.org",
    "https://irsafam.com",
    "https://zaban24.com/ielts",
    "https://www.ieltstoday.com",
    "https://ielts-rooz.ir",
    "https://www.ieltscity.com",
    "https://mohsen-ielts.com",
    "https://www.ieltstehran.com",
    "https://ieltsidp.com",
    "https://blog.faradars.org/ielts",
    "https://7learn.com/blog/ielts",
    "https://ili.ut.ac.ir/ielts"
]

# منابع بین‌المللی برای ترجمه
international_sources = [
    "https://www.ielts.org",
    "https://www.cambridgeenglish.org/exams-and-tests/ielts",
    "https://takeielts.britishcouncil.org",
    "https://ielts.idp.com"
]

# مترجم
translator = Translator(service_urls=['translate.googleapis.com'])

# راه‌اندازی مدل تشخیص سوالات
try:
    tokenizer = AutoTokenizer.from_pretrained("persiannlp/parsbert-qa-quest")
    model = AutoModelForSequenceClassification.from_pretrained("persiannlp/parsbert-qa-quest")
    question_detector = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1  # استفاده از CPU (-1) یا GPU (0)
    )
    print("✅ مدل parsBERT با موفقیت بارگذاری شد")
except Exception as e:
    print(f"⚠️ خطا در بارگذاری مدل parsBERT: {str(e)}")
    question_detector = None

# الگوهای پیشرفته سوالات
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
    'تحصیل', 'آزمون', 'زبان انگلیسی', 'ماژول'
]

def is_relevant(url, text):
    """بررسی ارتباط محتوا با آیلتس"""
    text = text.lower()
    url_check = any(kw in url.lower() for kw in ielts_keywords)
    content_check = any(kw in text[:500] for kw in ielts_keywords)
    
    if question_detector and len(text.split()) > 10:
        try:
            result = question_detector(text[:512])  # محدودیت طول مدل
            return result[0]['label'] == 'LABEL_1' or url_check or content_check
        except:
            pass
    
    return url_check or content_check

def translate_to_persian(text):
    """ترجمه محتوای انگلیسی به فارسی"""
    try:
        if text.strip():
            translated = translator.translate(text, src='en', dest='fa')
            return translated.text
        return text
    except Exception as e:
        print(f"خطا در ترجمه: {str(e)}")
        return text

def clean_text(text):
    """پاکسازی متن"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\b(?:https?|www)\S+', '', text)
    return text.strip()

def extract_links(url):
    """استخراج لینک‌های مرتبط"""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            text = a.get_text().lower()
            
            if any(kw in href or kw in text for kw in ielts_keywords):
                full_url = urljoin(url, href)
                if not any(ext in full_url for ext in ['.pdf', '.jpg', '.png']):
                    links.add(full_url)
        
        return list(links)[:10]  # محدودیت تعداد لینک
    except Exception as e:
        print(f"خطا در استخراج لینک‌ها از {url}: {str(e)}")
        return []

def scrape_page(url):
    """اسکرپ هوشمند محتوای صفحه"""
    try:
        response = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # حذف عناصر غیرمفید
        for tag in ['script', 'style', 'iframe', 'nav', 'footer', 'header', 
                   'form', 'button', 'input', 'select']:
            for element in soup(tag):
                element.decompose()
        
        # استخراج محتوای اصلی
        content = []
        for tag in ['article', 'main', 'section', 'div', 'p']:
            for element in soup.find_all(tag):
                text = clean_text(element.get_text())
                if len(text.split()) > 5 and is_relevant(url, text):
                    content.append(text)
        
        return ' '.join(content)
    except Exception as e:
        print(f"خطا در اسکرپ صفحه {url}: {str(e)}")
        return ""

def is_question(text):
    """تشخیص سوال با ترکیب الگوها و مدل NLP"""
    if len(text.split()) < 3:
        return False
    
    has_question_mark = '؟' in text or '?' in text
    pattern_check = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
    
    if question_detector:
        try:
            result = question_detector(text[:512])
            model_check = result[0]['label'] == 'LABEL_1'
            confidence = result[0]['score'] > 0.7
            return (model_check and confidence) or (pattern_check and has_question_mark)
        except Exception as e:
            print(f"خطا در تشخیص سوال: {str(e)}")
            return pattern_check and has_question_mark
    
    return pattern_check and has_question_mark

def generate_qa(text, source_type):
    """تولید هوشمند سوال-پاسخ"""
    qa_pairs = []
    
    # تقسیم متن به جملات
    sentences = [s.strip() for s in re.split(r'([؟?!\.]+)', text) if s.strip()]
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) and 
                re.match(r'^[؟?!\.]+$', sentences[i+1]) else '') 
                for i in range(0, len(sentences), 2)]
    
    for i in range(len(sentences)-1):
        current = sentences[i]
        next_sent = sentences[i+1] if i+1 < len(sentences) else ""
        
        if is_question(current) and len(next_sent.split()) > 3:
            if source_type == "international":
                current = translate_to_persian(current)
                next_sent = translate_to_persian(next_sent)
            
            qa_pairs.append({
                "question": current,
                "answer": next_sent,
                "confidence": question_detector(current[:512])[0]['score'] if question_detector else 0.9
            })
    
    return qa_pairs

def process_source(url, source_type):
    """پردازش هوشمند منبع"""
    links = extract_links(url)
    all_qa = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scrape_page, link): link for link in links}
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(links), 
                          desc=f"پردازش {url.split('//')[1][:20]}..."):
            content = future.result()
            if content:
                all_qa.extend(generate_qa(content, source_type))
    
    return all_qa

def main():
    """تابع اصلی"""
    print("\n🔍 شروع فرآیند جمع‌آوری داده با هوش مصنوعی...\n")
    
    dataset = {
        "metadata": {
            "version": "2.1",
            "model": "parsBERT-QA" if question_detector else "regex-only",
            "created_at": datetime.datetime.now().isoformat(),
            "sources": persian_sources + international_sources
        },
        "data": []
    }
    
    # پردازش منابع فارسی
    print("\n📚 در حال پردازش منابع فارسی...")
    for source in tqdm(persian_sources, desc="منابع فارسی"):
        dataset["data"].extend(process_source(source, "persian"))
    
    # پردازش منابع بین‌المللی
    print("\n🌍 در حال پردازش منابع بین‌المللی...")
    for source in tqdm(international_sources, desc="منابع خارجی"):
        dataset["data"].extend(process_source(source, "international"))
    
    # فیلتر نهایی
    print("\n🧹 در حال پالایش نهایی داده‌ها...")
    unique_qa = []
    seen = set()
    
    for item in tqdm(dataset["data"], desc="حذف تکراری‌ها"):
        key = (clean_text(item["question"]), clean_text(item["answer"]))
        if key not in seen and item.get("confidence", 1) > 0.6:
            seen.add(key)
            del item["confidence"]  # حذف فیلد موقت
            unique_qa.append(item)
    
    dataset["data"] = unique_qa
    dataset["metadata"]["total_qa"] = len(unique_qa)
    
    # ذخیره دیتاست نهایی
    output_file = 'ielts_ai_dataset_final.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ دیتاست نهایی با {len(unique_qa)} سوال-پاسخ در فایل '{output_file}' ذخیره شد.")
    print(f"⚙️ تنظیمات: مدل استفاده شده: {dataset['metadata']['model']}")

if __name__ == "__main__":
    main()