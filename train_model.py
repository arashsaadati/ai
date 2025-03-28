import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# 1. First make absolutely sure we're using the fast tokenizer
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    # Verify it's actually the fast version
    if not hasattr(tokenizer, '_tokenizer'):
        raise ValueError("Loaded tokenizer is not the fast version!")
except Exception as e:
    print(f"Error loading fast tokenizer: {e}")
    # Fallback to explicitly downloading the fast tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)

model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# 2. Enhanced dataset loading with more validation
def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    if "data" not in raw_data:
        raise ValueError("JSON file must contain a 'data' key")
    
    valid_data = []
    for item in raw_data["data"]:
        if not all(k in item for k in ['question', 'context', 'answer']):
            continue
        
        # More robust answer checking
        context = item['context']
        answer = item['answer']
        if not isinstance(answer, str) or not isinstance(context, str):
            continue
            
        if answer.strip() not in context:
            # Try case-insensitive search
            if answer.lower() not in context.lower():
                continue
        
        valid_data.append(item)
    
    if not valid_data:
        raise ValueError("No valid data found after validation")
    
    return Dataset.from_dict({
        'question': [item['question'] for item in valid_data],
        'context': [item['context'] for item in valid_data],
        'answer': [item['answer'] for item in valid_data]
    })

dataset = load_dataset('ielts_dataset.json')

# 3. Simplified preprocessing with better error handling
def preprocess_function(examples):
    try:
        # Tokenize with explicit settings for QA
        inputs = tokenizer(
            examples["question"],
            examples["context"],
            max_length=384,  # Slightly shorter for efficiency
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"  # Return PyTorch tensors directly
        )
        
        # Process answer positions
        start_positions = []
        end_positions = []
        
        for i in range(len(inputs["input_ids"])):
            # Get the original example index (accounts for overflow)
            sample_idx = inputs["overflow_to_sample_mapping"][i]
            answer = examples["answer"][sample_idx]
            context = examples["context"][sample_idx]
            
            # Find character positions
            start_char = context.find(answer)
            if start_char == -1:
                # Try case-insensitive search if exact match fails
                start_char = context.lower().find(answer.lower())
            
            if start_char == -1:
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            end_char = start_char + len(answer)
            
            # Find token positions
            sequence_ids = inputs.sequence_ids(i)
            context_start = sequence_ids.index(1)  # First context token
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)  # Last context token
            
            # Convert character positions to token positions
            offsets = inputs["offset_mapping"][i]
            start_token = context_start
            while start_token <= context_end and offsets[start_token][0] <= start_char:
                start_token += 1
            start_positions.append(start_token - 1)
            
            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= end_char:
                end_token -= 1
            end_positions.append(end_token + 1)
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise

# 4. Dataset preparation with error handling
try:
    train_test_split = dataset.train_test_split(test_size=0.1)
    tokenized_train = train_test_split["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=train_test_split["train"].column_names,
        batch_size=8  # Smaller batches if memory issues
    )
    tokenized_eval = train_test_split["test"].map(
        preprocess_function,
        batched=True,
        remove_columns=train_test_split["test"].column_names,
        batch_size=8
    )
except Exception as e:
    print(f"Error preparing datasets: {e}")
    raise

# 5. Training with more robust settings
training_args = TrainingArguments(
    output_dir="./ielts_model",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_num_workers=4,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# 6. Training with exception handling
try:
    print("Starting training...")
    trainer.train()
    print("Training completed successfully!")
    
    print("Saving model...")
    trainer.save_model("./ielts_model")
    tokenizer.save_pretrained("./ielts_model")
    print("Model saved successfully!")
    
except Exception as e:
    print(f"Error during training: {e}")
    # Attempt to save anyway if possible
    try:
        trainer.save_model("./ielts_model_partial")
        tokenizer.save_pretrained("./ielts_model_partial")
        print("Partial model saved to ielts_model_partial")
    except:
        print("Could not save partial model")
    raise