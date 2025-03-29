import requests
from bs4 import BeautifulSoup
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# قدم ۱: جمع‌آوری سوالات از اینترنت
def scrape_ielts_questions(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        questions = [item.get_text().strip() for item in soup.find_all('p') if '?' in item.get_text()]
        return questions
    except Exception as e:
        print(f"خطا توی اسکریپینگ: {e}")
        return []

# ذخیره سوالات
url = "https://gist.githubusercontent.com/arashsaadati/31a19a0269dcf9fb6f06a9faea67c5c4/raw/dc977571a62a5cbf832f5bae577fc6a89c96e8ce/ielts_dataset.json"  # عوض کن با URL واقعی
questions = scrape_ielts_questions(url)
if questions:
    with open('ielts_questions.json', 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    print(f"{len(questions)} سوال جمع شد و توی ielts_questions.json ذخیره شد.")
else:
    print("هیچ سوالی جمع نشد!")

# قدم ۲: دیتاست اولیه برای آموزش
training_data = [
    {"question": "هزینه آزمون چقدره؟", "answer": "هزینه آزمون آیلتس فعلاً ۱۳,۵۰۰,۰۰۰ تومان هست، ولی این قیمت به‌صورت علی‌الحسابه و اگه زمان اعلام تغییر قیمت بشه، باید مابه‌التفاوت رو پرداخت کنی."},
    {"question": "تاریخ آزمون کیه؟", "answer": "تاریخ آزمون بستگی به مرکز داره، لطفاً سایت رسمی رو چک کنید."},
    {"question": "مکان آزمون کجاست؟", "answer": "مکان آزمون توی شهرهای مختلف فرق داره، باید ثبت‌نام کنی تا ببینی."},
    {"question": "مدارک لازم چیه؟", "answer": "برای ثبت‌نام آیلتس به پاسپورت معتبر و یه عکس نیاز داری."}
]

# قدم ۳: آموزش مدل
questions = [item["question"] for item in training_data]
answers = [item["answer"] for item in training_data]

model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)
model.fit(questions, answers)

# قدم ۴: جواب دادن به سوالات
def get_answer(question):
    question = question.lower().strip()
    try:
        predicted_answer = model.predict([question])[0]
        return predicted_answer
    except:
        return "سوال رو متوجه نشدم، لطفاً در مورد آزمون آیلتس بپرسید!"

# قدم ۵: چت‌بات
def chat():
    print("سلام! سوالت رو بپرس (برای خروج بنویس 'خروج')")
    while True:
        user_input = input("سوال: ")
        if user_input == "خروج":
            print("خداحافظ!")
            break
        answer = get_answer(user_input)
        print("جواب: " + answer)

if __name__ == "__main__":
    chat()