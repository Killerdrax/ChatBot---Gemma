We are using Gemma-2b-it model. And Quantizing the model to save some memory and space, which helps in faster execution.

RAG(retrieval-augmented generation) is a simple method where we dont need to train or fine tune the model. Just pass the file to the model and it answers from it.

We are using embeddings so that the model can find the answers quickly and saves memory usage.



In the train folder I have created 3 codes where one is training the base model, the other is training a quantized model and the other is training a quantized model using LoRA method.
