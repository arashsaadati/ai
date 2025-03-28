import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import torch
from tqdm import tqdm

# 1. Use ParsBERT for best Persian support
model_name = "HooshvareLab/bert-fa-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. Load and validate dataset
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)["data"]

# 3. Manual validation with exact positions
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

# 4. Convert to Dataset
dataset = Dataset.from_pandas(pd.DataFrame(processed_data))

# 5. Improved preprocessing
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

        # If answer is not in this chunk
        if (offsets[context_start][0] > answer_end or 
            offsets[context_end][1] < answer_start):
            inputs["start_positions"].append(0)
            inputs["end_positions"].append(0)
        else:
            # Find start token
            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start:
                token_start += 1
            inputs["start_positions"].append(token_start - 1)

            # Find end token
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end:
                token_end -= 1
            inputs["end_positions"].append(token_end + 1)

    return inputs

# 6. Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 7. Training configuration
training_args = TrainingArguments(
    output_dir="./ielts_model_pro",
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True
)

# 8. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 9. Train and save
trainer.train()
trainer.save_model("./ielts_model_pro")
tokenizer.save_pretrained("./ielts_model_pro")