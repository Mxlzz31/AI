import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Step 1: Load a pre-trained multilingual model (XLM-R for sentiment analysis)
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 2: Load a multilingual dataset (e.g., Amazon reviews in multiple languages)
dataset = load_dataset("tweet_eval", "sentiment")
# Load English subset initially
# You can also specify other languages like "de" for German, "fr" for French, etc.

# Step 3: Preprocess and tokenize input data
def preprocess(text):
    # Tokenize input text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    return tokens

# Step 4: Perform sentiment analysis
def predict_sentiment(text):
    inputs = preprocess(text)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get sentiment probabilities
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = torch.argmax(probabilities, dim=-1).item()

    # Map to sentiment labels
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_labels[sentiment_score]

    return sentiment, probabilities[0].tolist()

# Test the system with multilingual input examples
example_texts = [
    "This product is great!",  # English
    "Este producto es excelente!",  # Spanish
    "Ce produit est fantastique!",  # French
    "Dieses Produkt ist schlecht.",  # German
    "这个产品很好！"  # Chinese
]

# Output predictions for each example
for text in example_texts:
    sentiment, prob = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment} | Probabilities: {prob}\n")