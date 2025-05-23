from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Load fine-tuned model
qa_pipeline = pipeline(
    "question-answering",
    model="./ielts_model_pro",
    tokenizer="./ielts_model_pro"
)

# Default context
DEFAULT_CONTEXT = """
آیلتس یه آزمون بین‌المللی برای سنجش مهارت‌های زبانه که شامل چهار بخش لیسنینگ، ریدینگ، رایتینگ و اسپیکینگ می‌شه.
هزینه آزمون حدود ۶ میلیون تومنه، ولی برای قیمت دقیق باید به سایت ایرسافام سر بزنی.
برای ثبت‌نام باید به سایت ایرسافام بری، فرم آنلاین رو پر کنی و وقت آزمونت رو رزرو کنی.
نمره آیلتس از ۰ تا ۹ هست و میانگین نمره چهار بخش، نمره کلت می‌شه.
برای رایتینگ باید تمرین کنی که ایده‌هات رو سریع سازمان‌دهی کنی و از لغات متنوع استفاده کنی.
مدت زمان آزمون حدود ۲ ساعت و ۴۵ دقیقه‌ست، به جز اسپیکینگ که جداگانه برگزار می‌شه.
"""

@app.route('/')
def index():
    return render_template('index.html', default_context=DEFAULT_CONTEXT)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        question = data.get('question')
        context = data.get('context', DEFAULT_CONTEXT)
        
        if not question:
            return jsonify({'error': 'لطفاً سوال خود را وارد کنید'}), 400
            
        result = qa_pipeline(question=question, context=context)
        return jsonify({
            'answer': result['answer'],
            'score': float(result['score']),
            'context_used': context if 'context' in data else 'default'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)