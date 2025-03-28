import json
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Load dataset
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    if "data" not in raw_data:
        raise ValueError("JSON file must contain a 'data' key")
    data = raw_data["data"]

# Convert to Dataset
dataset = Dataset.from_dict({
    'question': [item['question'] for item in data],
    'context': [item['context'] for item in data],
    'answer': [item['answer'] for item in data]
})

# Preprocessing function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = [a.strip() for a in examples["answer"]]

    #encodings = tokenizer(questions, contexts, truncation=True, padding=True, max_length=512)
    encodings = tokenizer(questions, contexts, truncation=True, padding=True, max_length=512, return_overflowing_tokens=True)

    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        context_tokens = tokenizer(contexts[i], truncation=True, max_length=512, return_offsets_mapping=True)
        answer_start_char = contexts[i].find(answers[i])
        
        if answer_start_char == -1:
            start_positions.append(0)
            end_positions.append(0)
        else:
            answer_end_char = answer_start_char + len(answers[i])
            for idx, (start_char, end_char) in enumerate(context_tokens["offset_mapping"]):
                if start_char <= answer_start_char < end_char:
                    start_positions.append(idx)
                    break
            for idx, (start_char, end_char) in enumerate(context_tokens["offset_mapping"]):
                if start_char < answer_end_char <= end_char:
                    end_positions.append(idx)
                    break
            else:
                end_positions.append(min(len(context_tokens["input_ids"]) - 1, start_positions[-1] + 1))

    encodings.update({
        "start_positions": start_positions,
        "end_positions": end_positions
    })
    return encodings

# Split and tokenize dataset
train_test_split = dataset.train_test_split(test_size=0.1)
tokenized_train_dataset = train_test_split['train'].map(preprocess_function, batched=True)
tokenized_eval_dataset = train_test_split['test'].map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ielts_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True  # Enable if using GPU
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

trainer.train()

# Save model
model.save_pretrained("./ielts_model")
tokenizer.save_pretrained("./ielts_model")
print("مدل با موفقیت آموزش داده شد و ذخیره شد!")