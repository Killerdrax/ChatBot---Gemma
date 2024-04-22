from llama_index.embeddings.fastembed import FastEmbedEmbedding
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
from llama_index.core import PromptTemplate
import torch
from huggingface_hub import HfFolder
import os
import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class QAAssistant:
    def __init__(self):
        HfFolder.save_token("YOUR_HF_TOKEN")
        self.documents = SimpleDirectoryReader("E:/data/").load_data()
        self.system_prompt = "Example: You are an assistant etc."
        self.context_window = 8192
        self.max_new_tokens = 256
        self.query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    def loginCredentials(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    def embedings(self):
        embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = embed_model
        Settings.chunk_size = 512

    def modeldetails(self):  #We check of the model is available locally and use that, else if quantize option is enabled, download the model and quantize it.
        model_path = r"E:\gemma\gemma-2b-it"
        quantized_model_path = r"E:\gemma\gemma-2b-it-quantized-4bit"

        if os.path.exists(quantized_model_path):
            print(f"Using saved quantized model from {quantized_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
            llm = HuggingFaceLLM(
                context_window=self.context_window,
                max_new_tokens=self.max_new_tokens,
                generate_kwargs={"temperature": 0.7, "do_sample": False},
                system_prompt=self.system_prompt,
                query_wrapper_prompt=self.query_wrapper_prompt,
                tokenizer=tokenizer,
                model=AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto"),
            )
        elif os.path.exists(model_path):
            print(f"Using saved model and quantizing from {model_path}")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")

            # Save the quantized model
            os.environ["HUGGINGFACE_HUB_CACHE"] = r"E:\gemma\gemma-cache"
            model.save_pretrained(quantized_model_path)
            tokenizer.save_pretrained(quantized_model_path)

            llm = HuggingFaceLLM(
                context_window=self.context_window,
                max_new_tokens=self.max_new_tokens,
                generate_kwargs={"temperature": 0.7, "do_sample": False},
                system_prompt=self.system_prompt,
                query_wrapper_prompt=self.query_wrapper_prompt,
                tokenizer=tokenizer,
                model=model,
            )
        else:
            print("Downloading model")
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", quantization_config=quantization_config, device_map="auto")

            # Save the quantized model
            os.environ["HUGGINGFACE_HUB_CACHE"] = r"E:\gemma\gemma-cache"
            model.save_pretrained(quantized_model_path)
            tokenizer.save_pretrained(quantized_model_path)

            llm = HuggingFaceLLM(
                context_window=self.context_window,
                max_new_tokens=self.max_new_tokens,
                generate_kwargs={"temperature": 0.7, "do_sample": False},
                system_prompt=self.system_prompt,
                query_wrapper_prompt=self.query_wrapper_prompt,
                tokenizer=tokenizer,
                model=model,
            )

        Settings.llm = llm
        Settings.chunk_size = 512
        index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = index.as_query_engine()

    def predict(self, input_text, history=[]):
        response = self.query_engine.query(input_text)
        return str(response)

    def chat(self):
        print("Welcome")
        print("Type 'exit' to quit.")

        history = []
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break

            response = self.predict(user_input, history)
            print("Assistant:", response)
            history.append((user_input, response))

if __name__ == "__main__":
    qa_assistant = QAAssistant()
    qa_assistant.loginCredentials()
    qa_assistant.embedings()
    qa_assistant.modeldetails()
    qa_assistant.chat()
