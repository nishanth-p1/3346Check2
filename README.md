# Unified Stock Price Predictor

A comprehensive stock price prediction system that combines:
- **Historical stock data** from Yahoo Finance (via yfinance)
- **News sentiment analysis** using BERT-based models
- **LSTM neural networks** for time series prediction
- **7-day price forecasts** with buy/sell recommendations

## Features

- 📊 Fetches real-time stock data from Yahoo Finance
- 📰 Scrapes and analyzes news headlines for sentiment
- 🤖 Trains LSTM models with integrated sentiment features
- 🔮 Predicts stock prices for the next 7 days
- 💡 Provides buy/sell/hold recommendations with confidence scores

## Project Structure

```
unified_stock_predictor/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── 1_data_collection.py      # Fetch stock data and news
├── 2_sentiment_analysis.py   # Analyze news sentiment
├── 3_data_preparation.py     # Merge stock + sentiment data
├── 4_model_training.py       # Train LSTM model
├── 5_prediction.py          # Make predictions and recommendations
├── data/
│   ├── news/                # News headlines and sentiment scores
│   └── stocks/              # Stock data and merged datasets
└── models/                  # Trained models and scalers
```

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd unified_stock_predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with PyTorch, install it separately:
   ```bash
   pip install torch torchvision torchaudio
   ```

## Usage

Run the scripts in order:

### Step 1: Data Collection
Fetch stock data and news headlines:
```bash
python 1_data_collection.py AAPL
```
Or for a different ticker:
```bash
python 1_data_collection.py TSLA
```

### Step 2: Sentiment Analysis
Analyze news headlines for sentiment:
```bash
python 2_sentiment_analysis.py AAPL
```

### Step 3: Data Preparation
Merge stock data with sentiment scores:
```bash
python 3_data_preparation.py AAPL
```

### Step 4: Model Training
Train the LSTM model:
```bash
python 4_model_training.py AAPL
```

Optional parameters:
```bash
python 4_model_training.py AAPL 100 50 32
# Arguments: ticker, look_back, epochs, batch_size
```

### Step 5: Prediction
Get 7-day predictions and buy/sell recommendation:
```bash
python 5_prediction.py AAPL
```

## Example Output

```
======================================================================
Stock Prediction for AAPL
======================================================================

Current Price: $175.43

Predicting next 7 days...

======================================================================
7-Day Price Predictions:
======================================================================
Date         Predicted Price   Change %     Change $    
----------------------------------------------------------------------
2024-11-21   $176.50          +0.61%       $+1.07
2024-11-22   $177.20          +1.01%       $+1.77
2024-11-23   $177.85          +1.38%       $+2.42
...

======================================================================
Trading Recommendation:
======================================================================
Recommendation: BUY
Confidence: 75.0%
Expected 7-day average change: +2.15%
Price trend (Day 1 to Day 7): +1.50%
Average predicted price: $179.20
Predicted price range: $176.50 - $180.10
======================================================================
```

## Trading Recommendations

The system provides the following recommendations:

- **BUY**: Predicted average price is >2% higher with positive trend
- **WEAK BUY**: Predicted average price is 1-2% higher
- **HOLD**: Predicted price change is within ±1%
- **WEAK SELL**: Predicted average price is 1-2% lower
- **SELL**: Predicted average price is >2% lower with negative trend

Confidence scores are calculated based on the magnitude of predicted price changes.

## Model Architecture

- **LSTM Layers**: 3 layers with 100 neurons each
- **Dropout**: 0.3 to prevent overfitting
- **Look-back Window**: 100 days (configurable)
- **Features**: Stock OHLCV data + 5 sentiment scores
- **Output**: Next day closing price

## Notes

1. **News Scraping**: The system uses Yahoo Finance RSS feeds. For more comprehensive news coverage, consider integrating additional sources like NewsAPI or Alpha Vantage.

2. **Sentiment Model**: Uses `nlptown/bert-base-multilingual-uncased-sentiment` which provides 5-class sentiment scores. For financial-specific sentiment, consider fine-tuning on financial news.

3. **Prediction Method**: Uses iterative prediction where each day's prediction feeds into the next. This can accumulate errors over longer horizons.

4. **Data Requirements**: The model needs at least 100 days of historical data to make predictions.

5. **Market Hours**: Predictions are for trading days. The system doesn't account for weekends/holidays in date calculations.

## Troubleshooting

- **"Model not found"**: Run `4_model_training.py` first
- **"No news data"**: News scraping may fail if RSS feeds are unavailable. The system will use neutral sentiment (0.2 for each class)
- **Memory errors**: Reduce batch size or look_back window in training
- **Slow sentiment analysis**: The BERT model is large. First run may take time to download (~600MB)

## License

This project combines code from:
- Stock-Price-Prediction-Using-LSTM
- Stock-Prediction-using-News-Info-Sentiment

Please refer to their respective licenses.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Disclaimer

This tool is for educational and research purposes only. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always do your own research and consult with financial advisors before making investment decisions.

