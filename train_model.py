import json
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from datasets import Dataset
import torch
from tqdm import tqdm

# 1. Use ParsBERT for Persian
model_name = "HooshvareLab/bert-fa-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. Load and validate dataset
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)["data"]

processed_data = []
for sample in raw_data:
    context = sample["context"]
    answer = sample["answer"]
    start_pos = context.find(answer)
    
    if start_pos == -1:
        print(f"❌ Answer not found: '{answer}' in:\n{context}")
        continue
        
    processed_data.append({
        "question": sample["question"],
        "context": context,
        "answer": answer,
        "answer_start": start_pos,
        "answer_end": start_pos + len(answer)
    })

print(f"✅ Valid samples: {len(processed_data)}/{len(raw_data)}")

# 3. Convert to Dataset and split
dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
split_dataset = dataset.train_test_split(test_size=0.2)

# 4. Improved preprocessing
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=256,
        truncation="only_second",
        stride=64,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    inputs["start_positions"] = []
    inputs["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer_start = examples["answer_start"][sample_idx]
        answer_end = examples["answer_end"][sample_idx]

        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if (offsets[context_start][0] > answer_end or 
            offsets[context_end][1] < answer_start):
            inputs["start_positions"].append(0)
            inputs["end_positions"].append(0)
        else:
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start:
                token_start += 1
            inputs["start_positions"].append(token_start - 1)

            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end:
                token_end -= 1
            inputs["end_positions"].append(token_end + 1)

    return inputs

# 5. Apply preprocessing to both splits
tokenized_train = split_dataset["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=split_dataset["train"].column_names
)

tokenized_eval = split_dataset["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=split_dataset["test"].column_names
)

# 6. Updated training configuration
training_args = TrainingArguments(
    output_dir="./ielts_model_pro",
    eval_strategy="steps",  # Changed from evaluation_strategy
    eval_steps=100,
    save_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True
)

# 7. Trainer with proper initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,  # Added eval dataset
    data_collator=default_data_collator,
    tokenizer=tokenizer  # Still accepted in current versions
)

# 8. Train and save
trainer.train()
trainer.save_model("./ielts_model_pro")
tokenizer.save_pretrained("./ielts_model_pro")
print("✅ Training completed successfully!")