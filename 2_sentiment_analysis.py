"""
Sentiment Analysis Script
Performs sentiment analysis on news headlines using a BERT model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

def load_sentiment_model():
    """Load pre-trained sentiment analysis model."""
    print("Loading sentiment analysis model (nlptown/bert-base-multilingual-uncased-sentiment)...")
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    print("Model loaded successfully!")
    return tokenizer, model

def calculate_daily_sentiment(headlines, tokenizer, model):
    """
    Calculate average sentiment for a day's headlines.
    Returns a list of 5 floats (probabilities for 1..5 stars).
    """
    # No headlines for that day -> neutral distribution
    if not headlines or len(headlines) == 0:
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    texts = [h.get("heading", "") for h in headlines if h.get("heading")]
    if not texts:
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    batch_size = 16
    all_scores = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            scores = logits.softmax(dim=1)  # shape: (batch, 5)
            all_scores.append(scores)

    if not all_scores:
        # Shouldn't normally happen, but be safe
        return [0.2, 0.2, 0.2, 0.2, 0.2]

    combined_scores = torch.cat(all_scores, dim=0)
    average_score = combined_scores.mean(dim=0).tolist()

    return average_score

def analyze_and_save_sentiment(ticker):
    """Analyze sentiment for all headlines and save daily average scores."""
    input_file = f"data/news/{ticker}_headlines.json"
    output_file = f"data/news/{ticker}_daily_scores.json"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run data collection first.")
        return

    print(f"Loading headlines from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loading sentiment model...")
    tokenizer, model = load_sentiment_model()

    result = {}
    total_days = len(data)
    processed = 0
    total_headlines_used = 0

    print(f"\nAnalyzing sentiment for {total_days} days...")
    print("-" * 60)

    for date in sorted(data.keys()):
        headlines = data[date]
        num_heads = len(headlines)
        total_headlines_used += num_heads

        avg_score = calculate_daily_sentiment(headlines, tokenizer, model)

        # Safety: ensure we always store a 5-element list
        if not isinstance(avg_score, list) or len(avg_score) != 5:
            avg_score = [0.2, 0.2, 0.2, 0.2, 0.2]

        result[date] = avg_score
        processed += 1

        if processed % 25 == 0 or processed == total_days:
            print(f"Processed {processed}/{total_days} days (headlines this day: {num_heads})")

    os.makedirs("data/news", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Sentiment analysis complete!")
    print(f"Results saved to {output_file}")
    print(f"Total days analyzed: {len(result)}")
    print(f"Total headlines used: {total_headlines_used}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys

    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()

    print(f"\n{'='*60}")
    print(f"Sentiment Analysis for {TICKER}")
    print(f"{'='*60}\n")

    analyze_and_save_sentiment(TICKER)
