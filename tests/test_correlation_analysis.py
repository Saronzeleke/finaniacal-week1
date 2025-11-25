"""
Test Task 3 implementation against criteria
"""

import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from task3_correlation_analysis import Task3CorrelationAnalysis

class TestTask3Criteria(unittest.TestCase):
    
    def test_date_normalization(self):
        """Test Criteria 2: Date normalization"""
        analyzer = Task3CorrelationAnalysis('dummy', 'dummy')
        analyzer.news_data = pd.DataFrame({
            'date': ['2024-01-01 10:30:00', '2024-01-02 14:45:00'],
            'headline': ['test1', 'test2']
        })
        analyzer.stock_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02'],
            'Close': [100, 101]
        })
        
        analyzer.normalize_dates()
        
        # Check if dates are normalized (no time component)
        self.assertEqual(str(analyzer.news_data['date'].dtype), 'datetime64[ns]')
        self.assertEqual(str(analyzer.stock_data['Date'].dtype), 'datetime64[ns]')
    
    def test_sentiment_analysis(self):
        """Test Criteria 3: Sentiment analysis"""
        analyzer = Task3CorrelationAnalysis('dummy', 'dummy')
        analyzer.news_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-01'],
            'headline': ['Great news!', 'Bad news!']
        })
        
        sentiment_data = analyzer.perform_sentiment_analysis()
        
        self.assertIn('avg_sentiment', sentiment_data.columns)
        self.assertIn('article_count', sentiment_data.columns)
    
    def test_daily_returns(self):
        """Test Criteria 3: Daily returns computation"""
        analyzer = Task3CorrelationAnalysis('dummy', 'dummy')
        analyzer.stock_data = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Close': [100, 102, 101]
        })
        
        returns_data = analyzer.compute_daily_returns()
        
        self.assertIn('daily_return', returns_data.columns)
        self.assertEqual(len(returns_data), 2)  # First row removed due to NaN

if __name__ == '__main__':
    unittest.main()