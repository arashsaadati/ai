import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin
import concurrent.futures
from tqdm import tqdm
from googletrans import Translator

# تنظیمات پیشرفته
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "fa-IR,fa;q=0.9"
}

# منابع فارسی معتبر
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

# الگوهای پیشرفته سوالات (گسترش یافته)
question_patterns = [
    # 1. سوالات عمومی
    r"(آیلتس\s*چی[ست]|چیست\??)",
    r"(معرفی\s*آزمون\s*آیلتس|آشنایی\s*با\s*آیلتس)\??",
    r"(کاربرد|مزایا|فایده)\s*مدرک\s*آیلتس\??",
    
    # 2. ثبت‌نام و هزینه
    r"(هزینه|قیمت|مبلغ)\s*(آزمون|آیلتس)\s*چقدر\??",
    r"(ثبت\s*نام|نام\s*نویسی)\s*چگونه\??",
    r"(مراکز\s*برگزاری|محل\s*آزمون)\s*کجاست\??",
    r"(زمان|تاریخ)\s*بندی\s*آزمون\s*چگونه\??",
    
    # 3. بخش‌های آزمون
    r"(بخش|ماژول)\s*(لیسنینگ|شنیداری)\s*چطور\??",
    r"(ریدینگ|خواندن)\s*چند\s*سوال\s*دارد\??",
    r"(رایتینگ|نوشتاری)\s*چند\s*تسک\s*دارد\??",
    r"(اسپیکینگ|مصاحبه)\s*چند\s*دقیقه\s*است\??",
    
    # 4. نمره‌دهی
    r"(نمره|امتیاز|باند)\s*چطور\s*محاسبه\??",
    r"(حداقل|حداکثر)\s*نمره\s*چقدر\??",
    r"(نمره\s*مورد\s*نیاز\s*دانشگاه)\??",
    
    # 5. آمادگی
    r"(آماده\s*سازی|آمادگی)\s*چند\s*ماه\??",
    r"(منابع|کتاب|منبع)\s*پیشنهادی\??",
    r"(کلاس|دوره)\s*آموزشی\s*پیشنهاد\s*می‌کنید\??",
    
    # 6. تکنیک‌ها
    r"(تکنیک|راهکار|نکته)\s*بخش\s*ریدینگ\??",
    r"(مدیریت\s*زمان\s*در\s*آزمون)\??",
    r"(اشتباهات\s*رایج\s*در\s*آیلتس)\??",
    
    # 7. مقایسه‌ای
    r"(تفاوت|فرق|مقایسه)\s*آیلتس\s*و\s*(تافل|PTE)\??",
    r"(کدام\s*سخت‌تر\s*است)\??",
    
    # 8. شرایط خاص
    r"(تکرار|مجدد)\s*آزمون\s*چند\s*بار\??",
    r"(اعتراض\s*به\s*نمره)\s*چگونه\??",
    r"(تسهیلات\s*ویژه\s*برای\s*معلولین)\??",
    
    # 9. ساختاری
    r"(چرا|علت|دلیل)\s*.*\s*آیلتس\??",
    r"(آیا|ایا)\s*.*\s*آیلتس\??",
    r"(چگونه|چطور)\s*.*\s*آیلتس\??",
    
    # 10. پیشرفته
    r"(مصاحبه\s*آزمون\s*چگونه\s*ارزیابی\s*می‌شود)\??",
    r"(معیارهای\s*نمره\s*دهی\s*رایتینگ)\??",
    r"(سوالات\s*رایج\s*مصاحبه\s*آیلتس)\??"
]

# الگوهای کمکی
supplemental_patterns = [
    r"^(آیا).*\?$",
    r".*(می‌شود|می‌کنید|دارید|هستید)\?$",
    r".*(کدام|چه|چگونه|چرا|کی|کجا)\s.*\?$"
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
    return url_check or content_check

def translate_to_persian(text):
    """ترجمه محتوای انگلیسی به فارسی"""
    try:
        if text.strip():
            translated = translator.translate(text, src='en', dest='fa')
            return translated.text
        return text
    except:
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
            if any(kw in href or kw in a.get_text().lower() for kw in ielts_keywords):
                full_url = urljoin(url, href)
                if not any(ext in full_url for ext in ['.pdf', '.jpg', '.png']):
                    links.add(full_url)
        return list(links)[:15]
    except:
        return []

def scrape_page(url):
    """اسکرپ محتوای صفحه"""
    try:
        response = requests.get(url, headers=headers, timeout=20)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # حذف عناصر غیرمفید
        for tag in ['script', 'style', 'iframe', 'nav', 'footer', 'header']:
            for element in soup(tag):
                element.decompose()
        
        # استخراج محتوای اصلی
        content = []
        for tag in ['p', 'div', 'article', 'section', 'main']:
            for element in soup.find_all(tag):
                text = clean_text(element.get_text())
                if len(text.split()) > 5 and is_relevant(url, text):
                    content.append(text)
        
        return ' '.join(content)
    except:
        return ""

def is_question(text):
    """تشخیص سوال بودن متن با الگوهای جدید"""
    main_check = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
    supplemental_check = any(re.search(pattern, text, re.IGNORECASE) for pattern in supplemental_patterns)
    has_question_mark = '؟' in text or '?' in text
    min_length = len(text.split()) >= 3
    
    return (main_check or supplemental_check) and has_question_mark and min_length

def generate_qa(text, source_type):
    """تولید سوال-پاسخ از متن"""
    qa_pairs = []
    sentences = [s.strip() for s in re.split(r'(؟|\?|!|\.)', text) if s.strip()]
    
    for i in range(len(sentences)-1):
        current = sentences[i] + ('؟' if '؟' in sentences[i] or '?' in sentences[i] else '')
        next_sent = sentences[i+1] if i+1 < len(sentences) else ""
        
        if is_question(current) and len(next_sent.split()) > 4:
            if source_type == "international":
                current = translate_to_persian(current)
                next_sent = translate_to_persian(next_sent)
            
            qa_pairs.append({
                "question": current,
                "answer": next_sent
            })
    
    return qa_pairs

def process_source(url, source_type):
    """پردازش کامل یک منبع"""
    links = extract_links(url)
    all_qa = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scrape_page, link): link for link in links}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(links), desc=f"پردازش {url}"):
            content = future.result()
            if content:
                all_qa.extend(generate_qa(content, source_type))
    
    return all_qa

def main():
    """تابع اصلی"""
    dataset = {"data": []}
    
    # پردازش منابع فارسی
    print("\nدر حال پردازش منابع فارسی...")
    for source in tqdm(persian_sources, desc="منابع فارسی"):
        dataset["data"].extend(process_source(source, "persian"))
    
    # پردازش منابع بین‌المللی
    print("\nدر حال پردازش منابع بین‌المللی...")
    for source in tqdm(international_sources, desc="منابع خارجی"):
        dataset["data"].extend(process_source(source, "international"))
    
    # حذف موارد تکراری
    unique_qa = []
    seen = set()
    for item in dataset["data"]:
        key = (item["question"], item["answer"])
        if key not in seen:
            seen.add(key)
            unique_qa.append(item)
    
    # ذخیره دیتاست نهایی
    with open('ielts_persian_dataset_v2.json', 'w', encoding='utf-8') as f:
        json.dump({"data": unique_qa}, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ دیتاست نهایی با {len(unique_qa)} سوال-پاسخ ذخیره شد.")

if __name__ == "__main__":
    main()