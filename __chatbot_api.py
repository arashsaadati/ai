from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# بارگذاری مدل فاین‌تیون شده
qa_pipeline = pipeline(
    "question-answering",
    model="./ielts_model_pro",
    tokenizer="./ielts_model_pro"
)

# متن مرجع پیش‌فرض (برای سوالاتی که نیاز به context دارن)
default_context = """
آیلتس یه آزمون بین‌المللی برای سنجش مهارت‌های زبانه که شامل چهار بخش لیسنینگ، ریدینگ، رایتینگ و اسپیکینگ می‌شه.
هزینه آزمون حدود ۶ میلیون تومنه، ولی برای قیمت دقیق باید به سایت ایرسافام سر بزنی.
برای ثبت‌نام باید به سایت ایرسافام بری، فرم آنلاین رو پر کنی و وقت آزمونت رو رزرو کنی.
نمره آیلتس از ۰ تا ۹ هست و میانگین نمره چهار بخش، نمره کلت می‌شه.
برای رایتینگ باید تمرین کنی که ایده‌هات رو سریع سازمان‌دهی کنی و از لغات متنوع استفاده کنی.
مدت زمان آزمون حدود ۲ ساعت و ۴۵ دقیقه‌ست، به جز اسپیکینگ که جداگانه برگزار می‌شه.
"""

# تابع گرفتن جواب
def get_answer(question, context=default_context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    context = data.get('context', default_context)  # اگه context فرستاده نشد، از پیش‌فرض استفاده کن
    if not question:
        return jsonify({'error': 'سوال رو وارد کن!'}), 400
    answer = get_answer(question, context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)