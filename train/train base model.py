import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
import json
import os
from huggingface_hub import HfFolder

# Define paths and model ID
HfFolder.save_token("YOUR_HF_TOKEN")
model_id = "google/gemma-2b-it"
output_dir = "E:/gemma/train"
cache_dir = os.path.join(output_dir, "cache") # Setting up cache directory


# Load model and tokenizer without quantization
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, torch_dtype=torch.float16, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token


# Function to format prompt and response
def formatting_func(prompt, response):
    return f"Question: {prompt}\nAnswer: {response}"


# Load data from train.json
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['instruction'] = df.apply(lambda row: formatting_func(row['prompt'], row['response']), axis=1)
    return Dataset.from_pandas(df[['instruction']])


train_dataset = load_data('train.json')
train_dataset = train_dataset.rename_column('instruction','text') #Renaming the column


# Configure training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # Reduce batch size
    gradient_accumulation_steps=4,
    warmup_steps=3,  # Changed to an integer value.
    max_steps=100,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,  # Enable FP16 for faster training
)

# Configure Data Collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Configure Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=data_collator,
)

model.config.use_cache = False
trainer.train()


# Save fine-tuned model and tokenizer
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Function to generate response
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
prompt = "Question: What are the Examples of B2B Payments?\nAnswer:"
print(generate_response(trainer.model, tokenizer, prompt))
