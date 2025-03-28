from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="./ielts_model_final",
    tokenizer="./ielts_model_final"
)

test_sample = {
    "context": "هزینه آزمون آیلتس حدود ۶ میلیون تومان است.",
    "question": "هزینه آزمون چقدر است؟"
}

result = qa_pipeline(test_sample)
print(f"Question: {test_sample['question']}")
print(f"Answer: {result['answer']} (Score: {result['score']:.4f})")