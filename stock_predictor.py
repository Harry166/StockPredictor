import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        # Use better hyperparameters for RandomForest
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.accuracy = 0
        self.mse = 0
        self.scaler = StandardScaler()
        
    def create_features(self, df):
        """Create technical indicators and features"""
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # High-Low ratio
        df['HL_Ratio'] = df['High'] / df['Low']
        
        # Price position within the day's range
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Volume-Price trend
        df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change']
        
        return df
    
    def fetch_data(self):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(self.ticker)
            df = stock.history(period="2y")  # 2 years of data
            
            if df.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            # Create features
            df = self.create_features(df)
            
            # Create target variable (next day's closing price)
            df['Target'] = df['Close'].shift(-1)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return None
    
    def train_model(self):
        """Train the stock prediction model"""
        df = self.fetch_data()
        
        if df is None or df.empty:
            return False
        
        # Get Reddit sentiment (using current sentiment for all historical data)
        print(f"Fetching Reddit sentiment for {self.ticker}...")
        reddit_sentiment = self.get_reddit_sentiment(self.ticker)
        print(f"Reddit sentiment: {reddit_sentiment}")
        
        # Add Reddit sentiment as a feature
        df['Reddit_Sentiment'] = reddit_sentiment
        
        # Select features
        feature_columns = ['Close', 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                          'Price_Change', 'Volume_Change', 'Volatility', 
                          'HL_Ratio', 'Price_Position', 'Volume_Price_Trend', 'Reddit_Sentiment']
        
        X = df[feature_columns]
        y = df['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        self.accuracy = r2_score(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)
        
        # Save model
        model_path = f'{self.ticker}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'accuracy': self.accuracy,
                'mse': self.mse,
                'feature_columns': feature_columns
            }, f)
        
        return True
    
    def predict_future_prices(self, days=730):  # 2 years = 730 days
        """Predict future stock prices for the next 2 years with improved volatility modeling"""
        model_path = f'{self.ticker}_model.pkl'
        
        if not os.path.exists(model_path):
            if not self.train_model():
                return None
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.accuracy = model_data['accuracy']
            self.mse = model_data['mse']
            feature_columns = model_data['feature_columns']
        
        # Get recent data
        df = self.fetch_data()
        if df is None or df.empty:
            return None
        
        # Get current price and calculate historical volatility
        current_price = df['Close'].iloc[-1]
        historical_volatility = df['Close'].pct_change().std()
        
        # Get current Reddit sentiment
        print(f"Fetching current Reddit sentiment for {self.ticker}...")
        current_reddit_sentiment = self.get_reddit_sentiment(self.ticker)
        print(f"Current Reddit sentiment: {current_reddit_sentiment}")
        
        # Generate future predictions with proper feature updating
        future_predictions = []
        future_dates = []
        
        # Initialize prediction variables
        last_close = current_price
        last_volume = df['Volume'].iloc[-1]
        price_history = df['Close'].tail(20).tolist()  # Keep last 20 days for MA calculation
        
        # Generate predictions for the next 2 years
        for i in range(days):
            # Create date for prediction
            pred_date = datetime.now() + timedelta(days=i+1)
            
            # Skip weekends (assuming stock market is closed)
            if pred_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue
            
            # Update moving averages dynamically
            ma_5 = np.mean(price_history[-5:]) if len(price_history) >= 5 else last_close
            ma_10 = np.mean(price_history[-10:]) if len(price_history) >= 10 else last_close
            ma_20 = np.mean(price_history[-20:]) if len(price_history) >= 20 else last_close
            
            # Calculate other technical indicators
            price_change = (last_close - price_history[-2]) / price_history[-2] if len(price_history) >= 2 else 0
            volume_change = 0.01  # Assume small volume change
            volatility = np.std(price_history[-10:]) if len(price_history) >= 10 else historical_volatility
            hl_ratio = 1.02  # Assume 2% high-low ratio
            price_position = 0.5  # Assume middle position
            volume_price_trend = last_volume * price_change
            
            # Create feature array (including Reddit sentiment)
            features = np.array([
                last_close,
                last_volume,
                ma_5,
                ma_10,
                ma_20,
                price_change,
                volume_change,
                volatility,
                hl_ratio,
                price_position,
                volume_price_trend,
                current_reddit_sentiment
            ])
            
            # Predict next price
            base_prediction = self.model.predict([features])[0]
            
            # Add realistic volatility and trend
            volatility_factor = np.random.normal(0, historical_volatility * 0.5)
            trend_factor = 0.0002 * i  # Small upward trend over time
            
            # Apply volatility and trend
            prediction = base_prediction * (1 + volatility_factor + trend_factor)
            
            # Ensure prediction doesn't go negative
            prediction = max(prediction, last_close * 0.5)
            
            future_predictions.append(prediction)
            future_dates.append(pred_date.strftime('%Y-%m-%d'))
            
            # Update variables for next iteration
            last_close = prediction
            price_history.append(prediction)
            if len(price_history) > 50:  # Keep only last 50 days
                price_history.pop(0)
            
            # Add some volume variation
            last_volume = last_volume * np.random.uniform(0.8, 1.2)
        
        # Calculate percentage change from current to final prediction
        final_price = future_predictions[-1] if future_predictions else current_price
        percentage_change = ((final_price - current_price) / current_price) * 100
        
        return {
            'ticker': self.ticker,
            'current_price': round(current_price, 2),
            'predicted_price': round(final_price, 2),
            'percentage_change': round(percentage_change, 2),
            'accuracy': round(self.accuracy * 100, 2),
            'mse': round(self.mse, 2),
            'future_predictions': [round(p, 2) for p in future_predictions],
            'future_dates': future_dates,
            'prediction_date': (datetime.now() + timedelta(days=730)).strftime('%Y-%m-%d')
        }
    
    def get_stock_info(self):
        """Get basic stock information"""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            return {
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A')
            }
        except:
            return {
                'name': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 'N/A'
            }

    def get_reddit_sentiment(self, ticker):
        """Fetch sentiment from Reddit for the given stock ticker"""
        try:
            reddit = praw.Reddit(
                client_id='U6lPM9dMzztYpJI6eEJeWg',
                client_secret='9vFC1M-zXLBS-NyjmNfembvGxShLvw',
                user_agent='StockPredictorApp'
            )

            analyzer = SentimentIntensityAnalyzer()
            sentiment_scores = []
            
            # List of relevant subreddits for stock discussion
            subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']
            
            for subreddit_name in subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Search for posts about the ticker
                    for submission in subreddit.search(ticker, limit=20, time_filter='day'):
                        # Check if post is recent (within 7 days)
                        if datetime.utcnow() - datetime.utcfromtimestamp(submission.created_utc) < timedelta(days=7):
                            # Analyze title and content
                            text = submission.title + " " + (submission.selftext if submission.selftext else "")
                            sentiment = analyzer.polarity_scores(text)
                            sentiment_scores.append(sentiment['compound'])
                            
                            # Also analyze top comments
                            submission.comments.replace_more(limit=0)
                            for comment in submission.comments[:5]:  # Top 5 comments
                                if hasattr(comment, 'body') and len(comment.body) > 10:
                                    comment_sentiment = analyzer.polarity_scores(comment.body)
                                    sentiment_scores.append(comment_sentiment['compound'])
                                    
                except Exception as e:
                    print(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue
            
            # Return average sentiment or 0 if no data found
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                print(f"Found {len(sentiment_scores)} sentiment data points")
                return avg_sentiment
            else:
                print("No recent Reddit posts found for this ticker")
                return 0
                
        except Exception as e:
            print(f"Error fetching Reddit sentiment: {e}")
            return 0

# Example usage and testing
if __name__ == "__main__":
    # Test the predictor
    predictor = StockPredictor("AAPL")
    result = predictor.predict_future_prices()
    
    if result:
        print("Stock Prediction Results:")
        print(f"Ticker: {result['ticker']}")
        print(f"Current Price: ${result['current_price']}")
        print(f"Predicted Price: ${result['predicted_price']}")
        print(f"Percentage Change: {result['percentage_change']}%")
        print(f"Model Accuracy: {result['accuracy']}%")
        print(f"Prediction Date: {result['prediction_date']}")
    else:
        print("Failed to generate prediction")
