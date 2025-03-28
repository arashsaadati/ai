import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# Load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Load and validate dataset
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    if "data" not in raw_data:
        raise ValueError("JSON file must contain a 'data' key")
    
    # Validate each item has required fields and answers are in contexts
    data = []
    for item in raw_data["data"]:
        if not all(k in item for k in ['question', 'context', 'answer']):
            print(f"Skipping item missing required fields: {item}")
            continue
        if item['answer'] not in item['context']:
            print(f"Skipping item where answer not in context: {item['answer']}")
            continue
        data.append(item)

    if not data:
        raise ValueError("No valid data found after validation")

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

    # Tokenize with fast tokenizer
    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",  # Truncate only the context
        stride=128,  # For handling overflow
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Get the mapping from overflowing tokens to original samples
    sample_map = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")
    
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        context = contexts[sample_idx]
        
        # Find character positions in original context
        start_char = context.find(answer)
        end_char = start_char + len(answer)
        
        # Initialize to 0 (will be used if answer not found)
        start_pos = 0
        end_pos = 0
        
        if start_char != -1:
            # Find the token positions
            sequence_ids = inputs.sequence_ids(i)
            
            # Find the start and end of the context (1 represents context tokens)
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
                
            token_end_index = len(sequence_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
                
            # If the answer is within the span
            if not (offsets[token_start_index][0] > end_char or offsets[token_end_index][1] < start_char):
                # Find start token index
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_pos = token_start_index - 1
                
                # Find end token index
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_pos = token_end_index + 1

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Split and tokenize dataset
train_test_split = dataset.train_test_split(test_size=0.1)
tokenized_train_dataset = train_test_split['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=train_test_split['train'].column_names
)
tokenized_eval_dataset = train_test_split['test'].map(
    preprocess_function,
    batched=True,
    remove_columns=train_test_split['test'].column_names
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ielts_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    logging_dir='./logs',
    logging_steps=50,
    report_to="tensorboard",
    warmup_ratio=0.1,
    save_total_limit=2
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

# Train and save model
print("Starting training...")
trainer.train()

print("Training complete. Saving model...")
model.save_pretrained("./ielts_model")
tokenizer.save_pretrained("./ielts_model")
print("Model saved successfully!")