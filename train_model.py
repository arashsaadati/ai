import json
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# بارگذاری مدل و توکنایزر
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# بارگذاری دیتاست به صورت دستی
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)["data"]


# تبدیل به فرمت Dataset
# اینجا مستقیم از data استفاده می‌کنیم چون خودش یه لیست از دیکشنری‌هاست
dataset = Dataset.from_dict({
    'question': [item['question'] for item in data],  # data خودش لیست دیکشنری‌هاست
    'context': [item['context'] for item in data],
    'answer': [item['answer'] for item in data]
})

# آماده‌سازی دیتا
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    answers = [a.strip() for a in examples["answer"]]

    encodings = tokenizer(questions, contexts, truncation=True, padding=True, max_length=512)

    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start = contexts[i].find(answers[i])
        end = start + len(answers[i])
        # اگه جواب تو context پیدا نشد، موقعیت صفر بذار (برای جلوگیری از خطا)
        if start == -1:
            start = 0
            end = 0
        start_positions.append(start)
        end_positions.append(end)

    encodings.update({
        "start_positions": start_positions,
        "end_positions": end_positions
    })
    return encodings

train_test_split = dataset.train_test_split(test_size=0.1)
tokenized_train_dataset = train_test_split['train'].map(preprocess_function, batched=True)

tokenized_eval_dataset = train_test_split['test'].map(preprocess_function, batched=True)


# تنظیمات آموزش
training_args = TrainingArguments(
    output_dir="./ielts_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# تربیت مدل
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,

)

trainer.train()

# ذخیره مدل
model.save_pretrained("./ielts_model")
tokenizer.save_pretrained("./ielts_model")
print("مدل با موفقیت آموزش داده شد و ذخیره شد!")
