from flask import Flask, render_template, request, jsonify
from stock_predictor import StockPredictor
import plotly.graph_objs as go
import plotly.utils
import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form.get('ticker', '').upper().strip()
        print(f"Processing request for ticker: {ticker}")
        
        if not ticker:
            return jsonify({'error': 'Please enter a stock ticker symbol'})
        
        # Create predictor instance
        predictor = StockPredictor(ticker)
        print(f"Created predictor for {ticker}")
        
        # Get prediction
        prediction_result = predictor.predict_future_prices()
        print(f"Prediction result: {prediction_result is not None}")
        
        if prediction_result is None:
            return jsonify({'error': f'Unable to fetch data for ticker: {ticker}. Please check if the ticker symbol is correct.'})
        
        # Get stock info
        stock_info = predictor.get_stock_info()
        print(f"Stock info obtained: {stock_info}")
        
        # Get Reddit sentiment
        reddit_sentiment = predictor.get_reddit_sentiment(ticker)
        print(f"Reddit sentiment: {reddit_sentiment}")
        
        # Get chart data with future predictions
        chart_data = get_stock_chart_data(ticker, prediction_result)
        print(f"Chart data created: {chart_data is not None}")
        
        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'stock_info': stock_info,
            'reddit_sentiment': reddit_sentiment,
            'chart_data': chart_data
        })
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred: {str(e)}'})

def get_stock_chart_data(ticker, prediction_result):
    """Get future prediction data for charting"""
    try:
        # Get recent historical data for context
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")  # 3 months of historical data
        
        if df.empty:
            return None
        
        # Prepare historical data
        historical_dates = df.index.strftime('%Y-%m-%d').tolist()
        historical_closes = df['Close'].round(2).tolist()
        
        # Get future predictions
        future_dates = prediction_result['future_dates']
        future_predictions = prediction_result['future_predictions']
        
        # Sample future predictions to avoid too many points (every 7th day)
        sampled_future_dates = future_dates[::7]  # Every 7th day
        sampled_future_predictions = future_predictions[::7]
        
        # Create Plotly chart
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_closes,
            mode='lines',
            name='Historical Price',
            line=dict(color='#007bff', width=2)
        ))
        
        # Add future predictions
        fig.add_trace(go.Scatter(
            x=sampled_future_dates,
            y=sampled_future_predictions,
            mode='lines',
            name='Future Predictions',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
        
        # Add current price marker
        current_date = datetime.now().strftime('%Y-%m-%d')
        fig.add_trace(go.Scatter(
            x=[current_date],
            y=[prediction_result['current_price']],
            mode='markers',
            name='Current Price',
            marker=dict(color='#28a745', size=10, symbol='circle')
        ))
        
        fig.update_layout(
            title=f'{ticker} Stock Price - Historical & 2-Year Predictions',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
