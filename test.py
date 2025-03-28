from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="./ielts_model",
    tokenizer="./ielts_model"
)

test_cases = [
    {
        "context": "هزینه آزمون آیلتس حدود ۶ میلیون تومان است. برای اطلاعات دقیق به سایت مراجعه کنید.",
        "question": "هزینه آزمون چقدر است؟"
    },
    {
        "context": "بخش رایتینگ آزمون آیلتس شامل دو تسک می‌شود.",
        "question": "رایتینگ آیلتس چند بخش دارد؟"
    }
]

for case in test_cases:
    result = qa_pipeline(case)
    print(f"Q: {case['question']}")
    print(f"A: {result['answer']}")
    print(f"Score: {result['score']:.2f}\n")