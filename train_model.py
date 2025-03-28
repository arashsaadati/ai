import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Use multilingual model for Persian
model_name = "distilbert-base-multilingual-cased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Load and verify dataset
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)["data"]

# Validate each sample
valid_data = []
for sample in data:
    context = sample["context"]
    answer = sample["answer"]
    if answer in context:
        valid_data.append(sample)
    else:
        print(f"Answer not found in context: {answer} | Context: {context}")

print(f"Using {len(valid_data)}/{len(data)} valid samples")

dataset = Dataset.from_dict({
    'question': [item['question'] for item in valid_data],
    'context': [item['context'] for item in valid_data],
    'answer': [item['answer'] for item in valid_data]
})

def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    inputs["start_positions"] = []
    inputs["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = examples["answer"][sample_idx]
        start_char = examples["context"][sample_idx].find(answer)
        end_char = start_char + len(answer)

        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If answer is not in the context, set to (0, 0)
        if start_char < offsets[context_start][0] or end_char > offsets[context_end][1]:
            inputs["start_positions"].append(0)
            inputs["end_positions"].append(0)
        else:
            # Find start and end token positions
            idx = context_start
            while idx <= context_end and offsets[idx][0] <= start_char:
                idx += 1
            inputs["start_positions"].append(idx - 1)

            idx = context_end
            while idx >= context_start and offsets[idx][1] >= end_char:
                idx -= 1
            inputs["end_positions"].append(idx + 1)

    return inputs

# Split and process
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"].map(preprocess_function, batched=True)
eval_dataset = dataset["test"].map(preprocess_function, batched=True)

# Training
training_args = TrainingArguments(
    output_dir="./ielts_model",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
trainer.save_model("./ielts_model")
tokenizer.save_pretrained("./ielts_model")