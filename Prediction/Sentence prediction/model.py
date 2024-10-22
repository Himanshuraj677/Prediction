import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class PredictiveTextModel:
    def __init__(self):
        # Load pre-trained model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def predict(self, input_text):
        # Encode input text and generate predictions
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate text
        output = self.model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

        # Decode generated text
        predicted_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return predicted_text

# Create an instance of the model for use in the Flask app
predictive_model = PredictiveTextModel()
