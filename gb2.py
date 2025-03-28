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

# تنظیمات هوشمند
class Config:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
            "Referer": "https://www.google.com/"
        }
        
        # منابع با قابلیت جایگزینی خودکار
        self.primary_sources = {
            "persian": [
                "https://irsafam.org",
                "https://zaban24.com/ielts",
                "https://gosafir.com/",
                "https://www.ielts4migration.com/",
                "https://zabanshenas.com/",
                "https://www.tehranpayment.com/",
                "https://maktabkhooneh.org/",
                "https://www.dorehsara.org/",
                "https://aryamcollege.com/",
                "https://fa.wikipedia.org/",
                "https://iran-europe.net/",
                "https://ieltslemon.com/fa",
                "https://go2tr.co/study/ielts",
                "https://www.ieltsadd.ir/",
                "https://www.idp.com/iran/ielts/",
                "https://sabzlearn.ir/",
                "https://zabanmehr.academy/",
                "https://afaqielts.com/",
                "https://tarjomic.com/",
                "https://armanienglish.com/",
                "https://englishturbo.com/",
                "https://www.proshotportal.com/",
                "https://diako.ac/",
                "https://www.ieltstehran.com",
                "https://ielts.idp.com",
                "https://www.ieltsiran.ir/",
                "hhttps://blog.faradars.org/search/?q=%D8%A2%DB%8C%D9%84%D8%AA%D8%B3#gsc.tab=0&gsc.q=%D8%A2%DB%8C%D9%84%D8%AA%D8%B3",
                "https://blog.faradars.org/category/english-language/",
                "https://7learn.com/blog/ielts",
                "https://ili.ut.ac.ir/ielts",
                "https://khaneyeielts.org/",
                "https://ielts2.com/",
                "https://www.ielts4migration.com/"
            ]
            # ,
            # "international": [
            #     "https://www.ielts.org",
            #     "https://ieltstheory.com/home/",
            #     "https://www.cambridgeenglish.org/exams-and-tests/ielts",
            #     "https://takeielts.britishcouncil.org",
            #     "https://ielts.idp.com"
            # ]
        }
        
        self.backup_sources = {
            "persian": [
                "https://fa.wikipedia.org/wiki/آیلتس",
                "https://7learn.com/blog/ielts",
                "https://blog.faradars.org/ielts"
            ],
            "international": [
                "https://ielts.idp.com",
                "https://www.cambridgeenglish.org/exams-and-tests/ielts"
            ]
        }
        
        self.current_sources = self.primary_sources.copy()
        
        # الگوهای سوالات با پشتیبانی گسترده
        self.question_patterns = [
            r"(آیلتس|تافل|PTE)\s*چی[ست]?\??",
            r"(چگونه|چطور|روش|طریقه|مراحل)\s.*\??",
            r"(هزینه|قیمت|مبلغ|شهریه)\s.*\??",
            r"(نمره|امتیاز|باند|نتیجه)\s.*\??",
            r"(منابع|کتاب|جزوه|نمونه سوال|تمرین)\s.*\??",
            r"(تفاوت|فرق|مقایسه|برتری)\s.*\??",
            r"(شرایط|نیازمندی|پیش نیاز|الزامات)\s.*\??",
            r"(آمادگی|آموزش|تمرین|کلاس)\s.*\??",
            r"(مدت|زمان|طول دوره|تاریخ)\s.*\??",
            r"(چرا|علت|دلیل|منظور)\s.*\??",
            r"(آیا|ایا|همینطور|میشه)\s.*\??",
            r"(کدام|چه|چگونه|چرا|کی|کجا|چقدر)\s.*\??",
            r".*\?(؟|\?|$)"
        ]
        
        self.keywords = [
            'آیلتس', 'ielts', 'ریدینگ', 'رایتینگ',
            'لیسنینگ', 'اسپیکینگ', 'نمره', 'مهاجرت',
            'تحصیل', 'آزمون', 'زبان انگلیسی', 'ماژول'
        ]
        
        self.setup_translator()
        self.setup_nlp_model()
    
    def setup_translator(self):
        """تنظیم مترجم با قابلیت fallback"""
        try:
            self.translator = Translator(service_urls=['translate.googleapis.com'])
            test_trans = self.translator.translate("hello", src='en', dest='fa').text
            assert test_trans == "سلام"
        except:
            print("Error on setting up translator: fallback to Google Translate")
            self.translator = None
    
    def setup_nlp_model(self):
        """تنظیم مدل NLP با قابلیت fallback"""
        try:
            self.question_detector = pipeline(
                "text-classification",
                model="HooshvareLab/bert-fa-base-uncased",
                device=-1
            )
            test_pred = self.question_detector("هزینه آزمون چقدر است؟")[0]
            assert test_pred['label'] == 'LABEL_1'
        except Exception as e:
            print(f"Error on setup NLP {str(e)}")
            self.question_detector = None
    
    def switch_to_backup(self):
        """جابجایی به منابع پشتیبان"""
        print("Switching to backup sources...")
        self.current_sources = self.backup_sources.copy()

config = Config()

# توابع اصلی با قابلیت تشخیص خودکار خطا
def smart_request(url, max_retries=3):
    """درخواست هوشمند با مدیریت خطا"""
    for i in range(max_retries):
        try:
            response = requests.get(url, headers=config.headers, timeout=15)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"error on get {url} (try {i+1}/{max_retries}): {str(e)}")
            if i == max_retries - 1:
                return None

def extract_links(url):
    """استخراج لینک‌های مرتبط با مدیریت خطا"""
    try:
        response = smart_request(url)
        if not response:
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            text = a.get_text().lower()
            
            if (any(kw in href or kw in text for kw in config.keywords) or
                (config.question_detector and len(text.split()) > 3 and 
                 config.question_detector(text[:100])[0]['label'] == 'LABEL_1')):
                
                full_url = urljoin(url, href)
                if not any(ext in full_url for ext in ['.pdf', '.jpg', '.png', '.docx']):
                    links.add(full_url)
        
        return list(links)[:15]
    except Exception as e:
        print(f"Error in extraction {url}: {str(e)}")
        return []

def clean_text(text):
    """پاکسازی متن با قابلیت انعطاف بیشتر"""
    if not text:
        return ""
        
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)
    text = re.sub(r'\b(?:https?|www|ftp)\S+', '', text)
    text = re.sub(r'[^\w\s.,;!?؟،]', '', text)
    return text.strip()

def is_question(text):
    """تشخیص سوال با ترکیب هوشمند روش‌ها"""
    if not text or len(text.split()) < 3:
        return False
    
    # بررسی الگوهای رجکس
    pattern_match = any(re.search(pattern, text, re.IGNORECASE) for pattern in config.question_patterns)
    
    # بررسی با مدل NLP اگر موجود باشد
    if config.question_detector:
        try:
            result = config.question_detector(text[:512])[0]
            model_match = result['label'] == 'LABEL_1' and result['score'] > 0.6
        except:
            model_match = False
    else:
        model_match = False
    
    # ترکیب نتایج
    return (pattern_match or model_match) and ('؟' in text or '?' in text)

def generate_qa(text, source_type):
    """تولید سوال-پاسخ با قابلیت انعطاف"""
    if not text:
        return []
    
    qa_pairs = []
    sentences = [s.strip() for s in re.split(r'([؟?!.])', text) if s.strip()]
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) and 
                re.match(r'^[؟?!.]$', sentences[i+1]) else '') 
                for i in range(0, len(sentences), 2)]
    
    for i in range(len(sentences)-1):
        current = sentences[i]
        next_sent = sentences[i+1] if i+1 < len(sentences) else ""
        
        if is_question(current) and len(next_sent.split()) > 2:
            if source_type == "international" and config.translator:
                try:
                    current = config.translator.translate(current, src='en', dest='fa').text
                    next_sent = config.translator.translate(next_sent, src='en', dest='fa').text
                except:
                    pass
            
            qa_pairs.append({
                "question": clean_text(current),
                "answer": clean_text(next_sent)
            })
    
    return qa_pairs

def process_source(url, source_type):
    """پردازش هوشمند یک منبع"""
    try:
        links = extract_links(url)
        if not links:
            print(f"No link find in {url}")
            return []
        
        all_qa = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(smart_request, link): link for link in links}
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(links), 
                             desc=f"Process {url[:30]}..."):
                link = futures[future]
                try:
                    response = future.result()
                    if response:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        content = ' '.join([clean_text(p.get_text()) 
                                          for p in soup.find_all(['p', 'div', 'article']) 
                                          if p.get_text().strip()])
                        all_qa.extend(generate_qa(content, source_type))
                except Exception as e:
                    print(f"Error in process {link}: {str(e)}")
        
        return all_qa
    except Exception as e:
        print(f"Error in process {url}: {str(e)}")
        return []

def main():
    print("\nStart scrabing\n")
    print(f"Current settings:")
    print(f"- Translate: {'Active' if config.translator else 'Deactive'}")
    print(f"- NLP Model: {'Active' if config.question_detector else 'Deactive'}")
    
    dataset = {
        "metadata": {
            "version": "3.0",
            "model": "bert-fa" if config.question_detector else "regex",
            "created_at": datetime.datetime.now().isoformat(),
            "sources": config.current_sources
        },
        "data": []
    }
    
    # پردازش منابع
    for lang, sources in config.current_sources.items():
        print(f"\nResource process {lang}...")
        for source in sources:
            qa_pairs = process_source(source, lang)
            if not qa_pairs and source in config.primary_sources[lang]:
                print(f"Main resource {source} not responding")
                config.switch_to_backup()
                qa_pairs = process_source(source, lang)
                
            dataset["data"].extend(qa_pairs)
            print(f"{len(qa_pairs)} question from {source} added")
    
    # حذف تکراری‌ها
    unique_qa = []
    seen = set()
    for item in dataset["data"]:
        key = (item["question"][:100], item["answer"][:100])
        if key not in seen:
            seen.add(key)
            unique_qa.append(item)
    
    dataset["data"] = unique_qa
    dataset["metadata"]["total_qa"] = len(unique_qa)
    
    # ذخیره دیتاست
    output_file = "ielts_smart_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"\nDataset with {len(unique_qa)} q/a saved")
    print(f"- File: {output_file}")
    print(f"- Model: {dataset['metadata']['model']}")
    print(f"- Resource: {len(dataset['metadata']['sources']['persian'])} persian, {len(dataset['metadata']['sources']['international'])} international")

if __name__ == "__main__":
    main()