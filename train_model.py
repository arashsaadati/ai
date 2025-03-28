import json
import pandas as pd  # <-- Add this import
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import torch
from tqdm import tqdm

# 1. Use XLM-RoBERTa for better multilingual support
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. Enhanced dataset validation
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)["data"]

processed_data = []
for sample in tqdm(raw_data, desc="Validating samples"):
    context = sample["context"].strip()
    answer = sample["answer"].strip()
    question = sample["question"].strip()
    
    # Find all answer occurrences
    start_indices = [i for i in range(len(context)) if context.startswith(answer, i)]
    
    if not start_indices:
        print(f"⚠️ Answer not found: '{answer}' in context: '{context[:50]}...'")
        continue
    
    processed_data.append({
        "question": question,
        "context": context,
        "answer": answer,
        "answer_start": start_indices[0],  # Use first occurrence
        "answer_end": start_indices[0] + len(answer)
    })

print(f"✅ Using {len(processed_data)}/{len(raw_data)} valid samples")

# 3. Create dataset using pandas DataFrame
dataset = Dataset.from_pandas(pd.DataFrame(processed_data))

def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    tokenized_inputs["start_positions"] = []
    tokenized_inputs["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_inputs["input_ids"][i]
        sequence_ids = tokenized_inputs.sequence_ids(i)

        # Get the sample context
        sample_index = sample_mapping[i]
        answer_start = examples["answer_start"][sample_index]
        answer_end = examples["answer_end"][sample_index]

        # Find start and end of context
        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        # If answer is out of span
        if offsets[context_start][0] > answer_end or offsets[context_end][1] < answer_start:
            tokenized_inputs["start_positions"].append(0)
            tokenized_inputs["end_positions"].append(0)
        else:
            # Find token positions
            start_token = context_start
            while start_token <= context_end and offsets[start_token][0] <= answer_start:
                start_token += 1
            tokenized_inputs["start_positions"].append(start_token - 1)

            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= answer_end:
                end_token -= 1
            tokenized_inputs["end_positions"].append(end_token + 1)

    return tokenized_inputs

# 4. Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 5. Split into train and eval sets
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)

# 6. Enhanced training configuration
training_args = TrainingArguments(
    output_dir="./ielts_model_enhanced",
    evaluation_strategy="steps",
    eval_steps=300,
    save_steps=300,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available()  # Enable mixed precision if GPU available
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
)

# 7. Train the model
trainer.train()

# 8. Save the final model
trainer.save_model("./ielts_model_final")
tokenizer.save_pretrained("./ielts_model_final")
print("✅ Training completed and model saved!")