from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="./ielts_model_pro",
    tokenizer="./ielts_model_pro"
)

sample = {
    "context": "هزینه آزمون آیلتس حدود ۶ میلیون تومان است.",
    "question": "هزینه آزمون چقدر است؟"
}

result = qa_pipeline(sample)
print(f"Question: {sample['question']}")
print(f"Answer: {result['answer']} (Score: {result['score']:.4f})")