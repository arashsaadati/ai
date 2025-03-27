import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin

# لیست URLهای پایه
base_urls = [
    "https://afarinesh.org",
    "https://irsafam.org",
    "https://ieltsadd.com",
    "https://gosafir.com",
    "https://fa.wikipedia.org",
    "https://www.ielts.org"
]

# هدرها برای جلوگیری از بلاک شدن توسط سایت‌ها
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# تابع برای پیدا کردن لینک‌های مرتبط در صفحه اصلی
def find_related_links(base_url):
    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = set()
        # پیدا کردن لینک‌هایی که به آیلتس مرتبط باشن
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if "ielts" in href.lower() or "آیلتس" in a_tag.get_text().lower():
                full_url = urljoin(base_url, href)
                links.add(full_url)
        return list(links)[:5]  # محدود کردن به ۵ لینک برای هر سایت
    except Exception as e:
        print(f"خطا در پیدا کردن لینک‌ها از {base_url}: {e}")
        return []

# تابع برای استخراج متن از صفحه وب
def scrape_page(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # استخراج متن از تگ‌های رایج
        content = soup.find_all(['p', 'div', 'article', 'section', 'li'])
        return [element.get_text().strip() for element in content if element.get_text().strip()]
    except Exception as e:
        print(f"خطا در اسکریپ {url}: {e}")
        return []

# تابع برای ساخت دیتاست
def build_dataset(texts):
    dataset = {"data": []}
    question_pattern = r"(چیه؟|چقدره؟|چطور|چه زمانی|چه ویژگی|چه تمرین|چطوریه؟|کجاست؟|کیه؟)"
    
    for text in texts:
        sentences = re.split(r'[؟.!]', text)
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:  # جملات خیلی کوتاه رو نادیده بگیر
                continue
            if re.search(question_pattern, sentence):
                question = sentence + "؟"
                answer = sentences[i + 1].strip() if i + 1 < len(sentences) and len(sentences[i + 1].strip()) > 5 else sentence
                dataset["data"].append({
                    "context": text,
                    "question": question,
                    "answer": answer
                })
    return dataset

# جمع‌آوری همه لینک‌ها و اسکریپ کردن
all_texts = []
for base_url in base_urls:
    print(f"در حال بررسی {base_url}...")
    related_links = find_related_links(base_url)
    for link in related_links:
        print(f"اسکریپ کردن {link}")
        texts = scrape_page(link)
        all_texts.extend(texts)

# ساخت دیتاست
dataset = build_dataset(all_texts)

# ذخیره دیتاست به صورت JSON
with open('ielts_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"دیتاست با موفقیت ذخیره شد! تعداد آیتم‌ها: {len(dataset['data'])}")
