from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset

# بارگذاری مدل و توکنایزر
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# بارگذاری دیتاست
dataset = load_dataset('json', data_files='ielts_dataset.json')['train']

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
        start_positions.append(start)
        end_positions.append(end)
    
    encodings.update({
        "start_positions": start_positions,
        "end_positions": end_positions
    })
    return encodings

tokenized_dataset = dataset.map(preprocess_function, batched=True)

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
    train_dataset=tokenized_dataset,
)

trainer.train()

# ذخیره مدل
model.save_pretrained("./ielts_model")
tokenizer.save_pretrained("./ielts_model")
print("مدل با موفقیت آموزش داده شد و ذخیره شد!")