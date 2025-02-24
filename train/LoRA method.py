import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, \
    BitsAndBytesConfig
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset
import json
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Define paths and model ID
quantized_model_path = "E:/gemma/gemma-2b-it-quantized-8bit/"  # Path to quantized model
trained_model_path = "E:/gemma/trained"  # Path to trained model
output_dir = "E:/gemma/"

# Check if trained model exists
if os.path.exists(trained_model_path) and os.listdir(trained_model_path):
    print("Loading trained model from:", trained_model_path)
    # Load trained model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(trained_model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
else:
    print("Training new model from quantized model")
    # Load quantized model and tokenizer
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(quantized_model_path, quantization_config=bnb_config,
                                                 torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)


    # Find linear layer names for Lora
    def find_all_linear_names(model):
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)


    modules = find_all_linear_names(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)


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
    train_dataset = train_dataset.rename_column('instruction', 'text')  # Renaming the column

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # increased
        warmup_steps=3,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        fp16=True,
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
    trainer.model.save_pretrained(trained_model_path)
    tokenizer.save_pretrained(trained_model_path)


print("Trained Model Found, skipping training")


# Function to generate response
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
prompt = "Question: What are the Examples of B2B Payments?\nAnswer:"
print(generate_response(model, tokenizer, prompt))

"""
Works, But its taking wayy too much time to train and my hardware is capped. Almost 3mins for a percentage
"""
