import json
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Load dataset
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)["data"]

# Convert to Dataset
dataset = Dataset.from_dict({
    'question': [item['question'] for item in data],
    'context': [item['context'] for item in data],
    'answer': [item['answer'] for item in data]
})

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = [a.strip() for a in examples["answer"]]

    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = contexts[sample_idx].find(answer)
        end_char = start_char + len(answer)

        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If answer is not fully in the context, label it as (0, 0)
        if start_char < offsets[context_start][0] or end_char > offsets[context_end][1]:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise find the start and end token indices
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.1)
tokenized_train_dataset = train_test_split['train'].map(preprocess_function, batched=True)
tokenized_eval_dataset = train_test_split['test'].map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ielts_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# Train and save
trainer.train()
trainer.save_model("./ielts_model")
tokenizer.save_pretrained("./ielts_model")
print("Model trained and saved successfully!")