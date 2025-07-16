import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
import pickle

class StockPredictor:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = LinearRegression()

    def fetch_data(self):
        data = yf.Ticker(self.ticker)
        df = data.history(period="5y")
        df['Tomorrow'] = df['Close'].shift(-1)
        df = df.dropna()
        return df

    def train(self):
        df = self.fetch_data()
        X = df[['Close', 'Volume']]
        y = df['Tomorrow']
        self.model.fit(X, y)
        with open(f'{self.ticker}_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self, recent_data):
        with open(f'{self.ticker}_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        prediction = self.model.predict([recent_data])
        return prediction[0]
