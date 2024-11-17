from transformers import BartForConditionalGeneration, BartTokenizer

# Load the pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text: str):
    # Encode the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary (using beam search for better results)
    summary_ids = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)

    # Decode the summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

text_to_summarize = input("Enter the text you want to summarize:")

summary = summarize_text(text_to_summarize)
print("Summary:", summary)
