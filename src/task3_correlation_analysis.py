"""
Task 3: Correlation Analysis between News Sentiment and Stock Returns
Author: Financial Analysis Project
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Analyzes correlation between news sentiment and stock price movements
    """
    
    def __init__(self, news_data_path, stock_data_path):
        """
        Initialize the correlation analyzer with data paths
        
        Args:
            news_data_path (str): Path to news headlines CSV
            stock_data_path (str): Path to stock data CSV
        """
        self.news_data_path = news_data_path
        self.stock_data_path = stock_data_path
        self.news_data = None
        self.stock_data = None
        self.sentiment_data = None
        self.correlation_results = {}
        
    def load_data(self):
        """
        Load and prepare news and stock data
        """
        try:
            # Load news data
            self.news_data = pd.read_csv(self.news_data_path)
            print(f"Loaded news data with {len(self.news_data)} records")
            
            # Load stock data
            self.stock_data = pd.read_csv(self.stock_data_path)
            print(f"Loaded stock data with {len(self.stock_data)} records")
            
            # Convert date columns to datetime
            self._normalize_dates()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _normalize_dates(self):
        """
        Normalize date formats between news and stock data
        """
        # Convert news date to datetime and extract date only (remove time)
        self.news_data['publication_date'] = pd.to_datetime(self.news_data['publication_date']).dt.normalize()
        
        # Convert stock date to datetime
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date']).dt.normalize()
        
        print("Date normalization completed")
        print(f"News date range: {self.news_data['publication_date'].min()} to {self.news_data['publication_date'].max()}")
        print(f"Stock date range: {self.stock_data['Date'].min()} to {self.stock_data['Date'].max()}")
    
    def analyze_sentiment(self, text_column='headline'):
        """
        Perform sentiment analysis on news headlines using TextBlob
        
        Args:
            text_column (str): Column name containing text to analyze
        """
        print("Performing sentiment analysis on news headlines...")
        
        def get_sentiment(text):
            """Calculate sentiment polarity for a given text"""
            try:
                if pd.isna(text) or text == '':
                    return 0.0
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0.0
        
        # Apply sentiment analysis
        self.news_data['sentiment'] = self.news_data[text_column].apply(get_sentiment)
        
        # Group by date and calculate daily average sentiment
        daily_sentiment = self.news_data.groupby('publication_date')['sentiment'].agg([
            ('avg_sentiment', 'mean'),
            ('sentiment_std', 'std'),
            ('article_count', 'count')
        ]).reset_index()
        
        self.sentiment_data = daily_sentiment
        print(f"Sentiment analysis completed for {len(daily_sentiment)} days")
        
        return daily_sentiment
    
    def compute_daily_returns(self, price_column='Close'):
        """
        Compute daily returns for stock data
        
        Args:
            price_column (str): Column name containing closing prices
            
        Returns:
            pd.DataFrame: Stock data with daily returns
        """
        print("Computing daily returns...")
        
        # Sort by date
        stock_data_sorted = self.stock_data.sort_values('Date').copy()
        
        # Calculate daily returns
        stock_data_sorted['daily_return'] = stock_data_sorted[price_column].pct_change() * 100
        
        # Calculate log returns (alternative method)
        stock_data_sorted['log_return'] = np.log(stock_data_sorted[price_column] / stock_data_sorted[price_column].shift(1)) * 100
        
        # Remove first row with NaN returns
        stock_data_sorted = stock_data_sorted.dropna(subset=['daily_return'])
        
        print(f"Daily returns computed for {len(stock_data_sorted)} trading days")
        
        return stock_data_sorted
    
    def merge_sentiment_returns(self):
        """
        Merge sentiment data with stock returns data by date
        
        Returns:
            pd.DataFrame: Merged dataset for correlation analysis
        """
        # Compute returns
        returns_data = self.compute_daily_returns()
        
        # Merge on date
        merged_data = pd.merge(
            self.sentiment_data,
            returns_data[['Date', 'daily_return', 'log_return', 'Close', 'Volume']],
            left_on='publication_date',
            right_on='Date',
            how='inner'
        )
        
        print(f"Merged dataset contains {len(merged_data)} matching days")
        
        return merged_data
    
    def calculate_correlation(self, merged_data):
        """
        Calculate Pearson correlation between sentiment and returns
        
        Args:
            merged_data (pd.DataFrame): Merged sentiment and returns data
            
        Returns:
            dict: Correlation results
        """
        print("Calculating Pearson correlation coefficients...")
        
        results = {}
        
        # Correlation between average sentiment and daily returns
        corr_daily, p_value_daily = pearsonr(
            merged_data['avg_sentiment'], 
            merged_data['daily_return']
        )
        
        # Correlation between average sentiment and log returns
        corr_log, p_value_log = pearsonr(
            merged_data['avg_sentiment'], 
            merged_data['log_return']
        )
        
        # Correlation with lagged sentiment (sentiment today vs returns tomorrow)
        merged_data['next_day_return'] = merged_data['daily_return'].shift(-1)
        merged_data_lagged = merged_data.dropna()
        
        if len(merged_data_lagged) > 0:
            corr_lagged, p_value_lagged = pearsonr(
                merged_data_lagged['avg_sentiment'], 
                merged_data_lagged['next_day_return']
            )
        else:
            corr_lagged, p_value_lagged = (np.nan, np.nan)
        
        results = {
            'daily_return_correlation': corr_daily,
            'daily_return_p_value': p_value_daily,
            'log_return_correlation': corr_log,
            'log_return_p_value': p_value_log,
            'lagged_correlation': corr_lagged,
            'lagged_p_value': p_value_lagged,
            'sample_size': len(merged_data)
        }
        
        self.correlation_results = results
        return results
    
    def visualize_correlation(self, merged_data, save_path=None):
        """
        Create visualizations for correlation analysis
        
        Args:
            merged_data (pd.DataFrame): Merged sentiment and returns data
            save_path (str): Path to save visualizations
        """
        print("Creating correlation visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('News Sentiment vs Stock Returns Correlation Analysis', fontsize=16)
        
        # Plot 1: Scatter plot - Sentiment vs Daily Returns
        axes[0, 0].scatter(merged_data['avg_sentiment'], merged_data['daily_return'], alpha=0.6)
        axes[0, 0].set_xlabel('Average Daily Sentiment')
        axes[0, 0].set_ylabel('Daily Returns (%)')
        axes[0, 0].set_title(f'Sentiment vs Daily Returns\nCorrelation: {self.correlation_results["daily_return_correlation"]:.3f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(merged_data['avg_sentiment'], merged_data['daily_return'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(merged_data['avg_sentiment'], p(merged_data['avg_sentiment']), "r--", alpha=0.8)
        
        # Plot 2: Time series of sentiment and returns
        ax2 = axes[0, 1]
        ax2.plot(merged_data['publication_date'], merged_data['avg_sentiment'], 
                label='Avg Sentiment', color='blue', alpha=0.7)
        ax2.set_ylabel('Sentiment Score', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax2_returns = ax2.twinx()
        ax2_returns.plot(merged_data['publication_date'], merged_data['daily_return'], 
                        label='Daily Returns', color='red', alpha=0.7)
        ax2_returns.set_ylabel('Daily Returns (%)', color='red')
        ax2_returns.tick_params(axis='y', labelcolor='red')
        
        ax2.set_xlabel('Date')
        ax2.set_title('Sentiment and Returns Over Time')
        
        # Plot 3: Distribution of sentiment scores
        axes[1, 0].hist(merged_data['avg_sentiment'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Average Daily Sentiment')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Daily Average Sentiment')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Correlation heatmap
        correlation_matrix = merged_data[['avg_sentiment', 'daily_return', 'log_return', 'article_count', 'Volume']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1, 1], square=True)
        axes[1, 1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {save_path}")
        
        plt.show()
    
    def generate_report(self):
        """
        Generate a comprehensive correlation analysis report
        """
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nSample Size: {self.correlation_results['sample_size']} matching days")
        
        print("\nPEARSON CORRELATION RESULTS:")
        print(f"Sentiment vs Daily Returns: {self.correlation_results['daily_return_correlation']:.4f}")
        print(f"P-value: {self.correlation_results['daily_return_p_value']:.4f}")
        
        print(f"\nSentiment vs Log Returns: {self.correlation_results['log_return_correlation']:.4f}")
        print(f"P-value: {self.correlation_results['log_return_p_value']:.4f}")
        
        print(f"\nSentiment vs Next Day Returns: {self.correlation_results['lagged_correlation']:.4f}")
        print(f"P-value: {self.correlation_results['lagged_p_value']:.4f}")
        
        print("\nINTERPRETATION:")
        corr_strength = abs(self.correlation_results['daily_return_correlation'])
        if corr_strength < 0.1:
            strength = "negligible"
        elif corr_strength < 0.3:
            strength = "weak"
        elif corr_strength < 0.5:
            strength = "moderate"
        else:
            strength = "strong"
        
        print(f"The correlation between news sentiment and stock returns is {strength}.")
        
        if self.correlation_results['daily_return_p_value'] < 0.05:
            print("The correlation is statistically significant (p < 0.05).")
        else:
            print("The correlation is not statistically significant (p >= 0.05).")

def main():
    """
    Main execution function for Task 3
    """
    print("Starting Task 3: Correlation Analysis")
    
    # Initialize analyzer
    analyzer = CorrelationAnalyzer(
        news_data_path='data/processed/news_data.csv',  # Adjust path as needed
        stock_data_path='data/processed/stock_data.csv'  # Adjust path as needed
    )
    
    try:
        # Load and prepare data
        analyzer.load_data()
        
        # Perform sentiment analysis
        sentiment_data = analyzer.analyze_sentiment()
        
        # Merge sentiment with returns
        merged_data = analyzer.merge_sentiment_returns()
        
        # Calculate correlation
        correlation_results = analyzer.calculate_correlation(merged_data)
        
        # Generate visualizations
        analyzer.visualize_correlation(merged_data, save_path='results/task3_correlation_analysis.png')
        
        # Generate report
        analyzer.generate_report()
        
        # Save results
        merged_data.to_csv('data/processed/sentiment_returns_merged.csv', index=False)
        
        print("\nTask 3 completed successfully!")
        
    except Exception as e:
        print(f"Error in Task 3 execution: {e}")
        raise

if __name__ == "__main__":
    main()