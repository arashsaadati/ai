import json
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# 1. Initialize tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# 2. Dataset loading with validation
def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    valid_data = []
    for item in raw_data["data"]:
        if not all(k in item for k in ['question', 'context', 'answer']):
            continue
        
        context = str(item['context'])
        answer = str(item['answer'])
        
        # Case-insensitive search if exact match fails
        if answer not in context and answer.lower() not in context.lower():
            continue
            
        valid_data.append({
            'question': str(item['question']),
            'context': context,
            'answer': answer
        })
    
    return Dataset.from_dict({
        'question': [item['question'] for item in valid_data],
        'context': [item['context'] for item in valid_data],
        'answer': [item['answer'] for item in valid_data]
    })

dataset = load_dataset('ielts_dataset.json')

# 3. Preprocessing with proper stride handling
def preprocess_function(examples):
    max_length = 384  # Reduced from 512 for better memory usage
    stride = 64  # Must be significantly smaller than max_length
    
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    # Initialize positions
    start_positions = []
    end_positions = []
    
    for i in range(len(inputs["input_ids"])):
        sample_idx = inputs["overflow_to_sample_mapping"][i]
        answer = examples["answer"][sample_idx]
        context = examples["context"][sample_idx]
        
        # Find character positions
        start_char = context.find(answer)
        if start_char == -1:
            start_char = context.lower().find(answer.lower())
        
        if start_char == -1:
            start_positions.append(0)
            end_positions.append(0)
            continue
            
        end_char = start_char + len(answer)
        
        # Get sequence ids and offsets
        sequence_ids = inputs.sequence_ids(i)
        offsets = inputs["offset_mapping"][i]
        
        # Find context span (1 represents context tokens)
        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
            
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1
            
        # If answer is outside the span
        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find start position
            token_idx = context_start
            while token_idx <= context_end and offsets[token_idx][0] <= start_char:
                token_idx += 1
            start_positions.append(token_idx - 1)
            
            # Find end position
            token_idx = context_end
            while token_idx >= context_start and offsets[token_idx][1] >= end_char:
                token_idx -= 1
            end_positions.append(token_idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    
    # Convert to lists to avoid tensor issues
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "start_positions": start_positions,
        "end_positions": end_positions
    }

# 4. Dataset preparation
train_test_split = dataset.train_test_split(test_size=0.1)
tokenized_train = train_test_split["train"].map(
    preprocess_function,
    batched=True,
    remove_columns=train_test_split["train"].column_names,
    batch_size=4
)
tokenized_eval = train_test_split["test"].map(
    preprocess_function,
    batched=True,
    remove_columns=train_test_split["test"].column_names,
    batch_size=4
)

# 5. Training configuration
training_args = TrainingArguments(
    output_dir="./ielts_model",
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# 6. Training execution
print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model("./ielts_model_final")
tokenizer.save_pretrained("./ielts_model_final")
print("Training complete!")