from transformers import pipeline

class Inference:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        """
        Initialize the open-source language model.
        """
        print(f"Loading language model: {model_name}...")
        self.generator = pipeline("text-generation", model=model_name, device="cpu")  # Use GPU if available
        print("Language model loaded successfully.")

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using the language model.
        """
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        response = self.generator(prompt, max_length=200, num_return_sequences=1)
        return response[0]["generated_text"]