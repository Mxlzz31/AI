from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def classify_sentence(sentence: str):
    result = classifier(sentence)
    label = result[0]['label']
    score = result[0]['score']

    sentiment = "Neutral" if label == "LABEL_1" else "Positive" if label == "LABEL_2" else "Negative"

    return sentiment, score

while True:
    sentence = input("Enter a sentence to classify (or type 'exit' to quit): ")
    if sentence.lower() == 'exit':
        break

    sentiment, score = classify_sentence(sentence)
    print(f"Sentence: {sentence}")
    print(f"Classification: {sentiment}, Score: {score:.2f}\n")
