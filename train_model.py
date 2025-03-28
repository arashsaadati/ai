import json
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Load and validate dataset
with open('ielts_dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    if "data" not in raw_data:
        raise ValueError("JSON file must contain a 'data' key")
    
    # Validate each item has required fields
    data = []
    for item in raw_data["data"]:
        if not all(k in item for k in ['question', 'context', 'answer']):
            continue
        if item['answer'] not in item['context']:
            continue
        data.append(item)

dataset = Dataset.from_dict({
    'question': [item['question'] for item in data],
    'context': [item['context'] for item in data],
    'answer': [item['answer'] for item in data]
})

# Improved preprocessing function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    answers = [a.strip() for a in examples["answer"]]

    # Tokenize with proper truncation strategy for QA
    inputs = tokenizer(
        questions,
        contexts,
        max_length=512,
        truncation="only_second",  # This truncates only the context (second sequence)
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
        
        if start_char == -1:
            # Answer not found in context (shouldn't happen if data is clean)
            start_positions.append(0)
            end_positions.append(0)
            continue
            
        # Find the token positions
        sequence_ids = inputs.sequence_ids(i)
        
        # Find the start and end of the context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1
            
        token_end_index = len(sequence_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1
            
        # If the answer is out of the span, set to 0
        if offsets[token_start_index][0] > end_char or offsets[token_end_index][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise find the start and end token indices
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)
            
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Split and tokenize dataset
train_test_split = dataset.train_test_split(test_size=0.1)
tokenized_train_dataset = train_test_split['train'].map(preprocess_function, batched=True, remove_columns=train_test_split['train'].column_names)
tokenized_eval_dataset = train_test_split['test'].map(preprocess_function, batched=True, remove_columns=train_test_split['test'].column_names)

# Enhanced training arguments
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
    report_to="tensorboard"
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
print("Model trained and saved successfully!")