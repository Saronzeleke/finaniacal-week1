"""
Test cases for Task 3 Correlation Analysis
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from task3_correlation_analysis import CorrelationAnalyzer

class TestCorrelationAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Create sample news data
        self.sample_news = pd.DataFrame({
            'headline': [
                'Great company reports amazing profits',
                'Market crashes due to bad news',
                'Company announces new product',
                'Economic concerns rise',
                'Positive outlook for tech sector'
            ],
            'publication_date': pd.to_datetime([
                '2024-01-01', '2024-01-02', '2024-01-03', 
                '2024-01-04', '2024-01-05'
            ])
        })
        
        # Create sample stock data
        self.sample_stock = pd.DataFrame({
            'Date': pd.to_datetime([
                '2024-01-01', '2024-01-02', '2024-01-03', 
                '2024-01-04', '2024-01-05'
            ]),
            'Close': [100, 95, 98, 96, 102]
        })
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        analyzer = CorrelationAnalyzer('dummy_path1', 'dummy_path2')
        analyzer.news_data = self.sample_news
        
        sentiment_data = analyzer.analyze_sentiment()
        
        self.assertIn('avg_sentiment', sentiment_data.columns)
        self.assertIn('article_count', sentiment_data.columns)
        self.assertEqual(len(sentiment_data), 5)
    
    def test_daily_returns_calculation(self):
        """Test daily returns calculation"""
        analyzer = CorrelationAnalyzer('dummy_path1', 'dummy_path2')
        analyzer.stock_data = self.sample_stock
        
        returns_data = analyzer.compute_daily_returns()
        
        self.assertIn('daily_return', returns_data.columns)
        self.assertIn('log_return', returns_data.columns)
        # Should have 4 rows after removing first NaN
        self.assertEqual(len(returns_data), 4)

if __name__ == '__main__':
    unittest.main()