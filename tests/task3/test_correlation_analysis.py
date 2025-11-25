"""
Task 3: Correlation Analysis
Implementation strictly based on assessment criteria:
1. Normalizing dates between news and stock data
2. Performing sentiment analysis on news headlines
3. Computing daily returns
4. Calculating Pearson correlation coefficient
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr

class Task3CorrelationAnalysis:
    """
    Implements Task 3 requirements for correlation analysis
    """
    
    def __init__(self, news_path, stock_path):
        self.news_path = news_path
        self.stock_path = stock_path
        self.news_data = None
        self.stock_data = None
        
    def load_and_prepare_data(self):
        """Load and prepare both datasets"""
        # Load news data
        self.news_data = pd.read_csv(self.news_path)
        # Load stock data  
        self.stock_data = pd.read_csv(self.stock_path)
        
        print(f"Loaded {len(self.news_data)} news records")
        print(f"Loaded {len(self.stock_data)} stock records")
        
    def normalize_dates(self):
        """
        CRITERIA 2: Normalizing dates between news and stock data
        """
        # Convert news date to datetime and normalize
        self.news_data['date'] = pd.to_datetime(self.news_data['date']).dt.normalize()
        # Convert stock date to datetime and normalize
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date']).dt.normalize()
        
        print("Date normalization completed")
        
    def perform_sentiment_analysis(self):
        """
        CRITERIA 3: Performing sentiment analysis on news headlines
        """
        def calculate_sentiment(text):
            """Calculate sentiment polarity using TextBlob"""
            try:
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0.0
        
        # Apply sentiment analysis to headlines
        self.news_data['sentiment'] = self.news_data['headline'].apply(calculate_sentiment)
        
        # Calculate daily average sentiment
        daily_sentiment = self.news_data.groupby('date')['sentiment'].agg([
            ('avg_sentiment', 'mean'),
            ('article_count', 'count')
        ]).reset_index()
        
        print(f"Sentiment analysis completed for {len(daily_sentiment)} days")
        return daily_sentiment
    
    def compute_daily_returns(self):
        """
        CRITERIA 3: Computing daily returns
        """
        # Sort by date and calculate percentage returns
        self.stock_data = self.stock_data.sort_values('Date')
        self.stock_data['daily_return'] = self.stock_data['Close'].pct_change() * 100
        
        # Remove first row with NaN
        returns_data = self.stock_data.dropna(subset=['daily_return'])
        
        print(f"Daily returns computed for {len(returns_data)} trading days")
        return returns_data
    
    def calculate_pearson_correlation(self, sentiment_data, returns_data):
        """
        CRITERIA 3: Calculating Pearson correlation coefficient
        """
        # Merge sentiment and returns data on date
        merged_data = pd.merge(
            sentiment_data,
            returns_data[['Date', 'daily_return']],
            left_on='date',
            right_on='Date',
            how='inner'
        )
        
        print(f"Merged dataset: {len(merged_data)} matching days")
        
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(
            merged_data['avg_sentiment'], 
            merged_data['daily_return']
        )
        
        results = {
            'pearson_correlation': correlation,
            'p_value': p_value,
            'sample_size': len(merged_data),
            'merged_data': merged_data
        }
        
        return results
    
    def execute_full_analysis(self):
        """
        Execute complete Task 3 analysis based on criteria
        """
        print("=== TASK 3: CORRELATION ANALYSIS ===")
        
        # Step 1: Load data
        self.load_and_prepare_data()
        
        # Step 2: Normalize dates (Criteria 2)
        self.normalize_dates()
        
        # Step 3: Perform sentiment analysis (Criteria 3)
        sentiment_data = self.perform_sentiment_analysis()
        
        # Step 4: Compute daily returns (Criteria 3)  
        returns_data = self.compute_daily_returns()
        
        # Step 5: Calculate Pearson correlation (Criteria 3)
        correlation_results = self.calculate_pearson_correlation(sentiment_data, returns_data)
        
        # Display results
        self.display_results(correlation_results)
        
        return correlation_results
    
    def display_results(self, results):
        """Display correlation results"""
        print("\n" + "="*50)
        print("TASK 3 RESULTS")
        print("="*50)
        print(f"Sample Size: {results['sample_size']} matching days")
        print(f"Pearson Correlation Coefficient: {results['pearson_correlation']:.4f}")
        print(f"P-value: {results['p_value']:.4f}")
        
        # Interpretation
        if results['p_value'] < 0.05:
            significance = "Statistically Significant"
        else:
            significance = "Not Statistically Significant"
            
        print(f"Statistical Significance: {significance}")
        print("="*50)

def main():
    """Main execution function"""
    analyzer = Task3CorrelationAnalysis(
        news_path='data/raw_analyst_ratings.csv',
        stock_path='data/NVDA.csv'
    )
    
    results = analyzer.execute_full_analysis()
    
    # Save results
    results['merged_data'].to_csv('data/processed/task3_correlation_data.csv', index=False)
    print("Results saved to 'data/processed/task3_correlation_data.csv'")

if __name__ == "__main__":
    main()