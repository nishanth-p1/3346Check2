"""
Data Preparation Script
Merges stock data with sentiment scores
"""

import pandas as pd
import json
import numpy as np
import os


def merge_stock_and_sentiment(ticker):
    """Merge stock data with sentiment scores"""
    # Load stock data
    stock_path = f'data/stocks/{ticker}_data.csv'
    if not os.path.exists(stock_path):
        print(f"Error: {stock_path} not found. Please run data collection first.")
        return None
    
    print(f"Loading stock data from {stock_path}...")
    stock_df = pd.read_csv(stock_path)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # Load sentiment scores
    sentiment_path = f'data/news/{ticker}_daily_scores.json'
    if not os.path.exists(sentiment_path):
        print(f"Error: {sentiment_path} not found. Please run sentiment analysis first.")
        return None
    
    print(f"Loading sentiment scores from {sentiment_path}...")
    with open(sentiment_path, 'r', encoding='utf-8') as f:
        sentiment_scores = json.load(f)
    
    # Convert JSON to DataFrame
    sentiment_df = pd.DataFrame([
        [pd.to_datetime(date)] + scores
        for date, scores in sentiment_scores.items()
    ], columns=['Date', 'sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4', 'sentiment_5'])
    
    # Merge on date
    print("Merging stock data with sentiment scores...")
    merged_df = pd.merge(stock_df, sentiment_df, on='Date', how='left')
    
    # Fill missing sentiment values with neutral defaults
    sentiment_cols = ['sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4', 'sentiment_5']
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0.2)
    
    # Drop unnecessary columns if exist
    if 'Adj Close' in merged_df.columns:
        merged_df = merged_df.drop(columns=['Adj Close'])
    
    # Sort by date
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    # Remove invalid rows
    merged_df = merged_df.dropna(subset=['Date'])
    merged_df = merged_df[pd.to_numeric(merged_df['Close'], errors='coerce').notna()]

    # Save merged data
    output_path = f'data/stocks/{ticker}_merged.csv'
    os.makedirs('data/stocks', exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Merged data saved to {output_path}")
    print(f"Shape: {merged_df.shape}")
    print(f"\nColumns: {list(merged_df.columns)}")
    print(f"\nFirst few rows:")
    print(merged_df.head())
    print(f"\nLast few rows:")
    print(merged_df.tail())
    print(f"{'='*60}\n")
    
    return merged_df


# def merge_stock_and_sentiment(ticker):
#     """Merge stock data with sentiment scores"""
#     # Load stock data
#     stock_path = f'data/stocks/{ticker}_data.csv'
#     if not os.path.exists(stock_path):
#         print(f"Error: {stock_path} not found. Please run data collection first.")
#         return None
    
#     print(f"Loading stock data from {stock_path}...")
#     stock_df = pd.read_csv(stock_path)
#     stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
#     # Load sentiment scores
#     sentiment_path = f'data/news/{ticker}_daily_scores.json'
#     if not os.path.exists(sentiment_path):
#         print(f"Error: {sentiment_path} not found. Please run sentiment analysis first.")
#         return None
    
#     print(f"Loading sentiment scores from {sentiment_path}...")
#     with open(sentiment_path, 'r', encoding='utf-8') as f:
#         sentiment_scores = json.load(f)
    
#     # Convert sentiment to DataFrame
#     sentiment_df = pd.DataFrame(
#         list(sentiment_scores.items()), 
#         columns=['jsonDate', 'sentiment']
#     )
#     sentiment_df['date'] = pd.to_datetime(sentiment_df['jsonDate'], format='%Y-%m-%d')
    
#     # Merge on date
#     print("Merging stock data with sentiment scores...")
#     merged_df = pd.merge(stock_df, sentiment_df, left_on='Date', right_on='date', how='left')
    
#     # Expand sentiment scores into separate columns
#     sentiment_columns = pd.DataFrame(
#         merged_df['sentiment'].tolist(),
#         columns=['sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4', 'sentiment_5']
#     )
    
#     # Fill NaN sentiment with neutral values (0.2 for each class)
#     sentiment_columns = sentiment_columns.fillna(0.2)
    
#     # Combine data
#     merged_df = pd.concat([merged_df, sentiment_columns], axis=1)
    
#     # Drop unnecessary columns
#     columns_to_drop = ['sentiment', 'jsonDate', 'date']
#     if 'Adj Close' in merged_df.columns:
#         columns_to_drop.append('Adj Close')
    
#     merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])
    
#     # Forward fill missing sentiment values
#     sentiment_cols = ['sentiment_1', 'sentiment_2', 'sentiment_3', 'sentiment_4', 'sentiment_5']
#     for col in sentiment_cols:
#         if col in merged_df.columns:
#             merged_df[col] = merged_df[col].ffill().fillna(0.2)
    
#     # Sort by date
#     merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
#     # Save merged data
#     output_path = f'data/stocks/{ticker}_merged.csv'
#     os.makedirs('data/stocks', exist_ok=True)
#     merged_df.to_csv(output_path, index=False)
    
#     print(f"\n{'='*60}")
#     print(f"Merged data saved to {output_path}")
#     print(f"Shape: {merged_df.shape}")
#     print(f"\nColumns: {list(merged_df.columns)}")
#     print(f"\nFirst few rows:")
#     print(merged_df.head())
#     print(f"\nLast few rows:")
#     print(merged_df.tail())
#     print(f"{'='*60}\n")
    
#     return merged_df

if __name__ == "__main__":
    import sys
    
    # Default ticker
    TICKER = "AAPL"
    if len(sys.argv) > 1:
        TICKER = sys.argv[1].upper()
    
    print(f"\n{'='*60}")
    print(f"Data Preparation for {TICKER}")
    print(f"{'='*60}\n")
    
    merge_stock_and_sentiment(TICKER)

