import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import DataLoader, EDAAnalyzer
from src.text_analyzer import TextAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EDAVisualizer:
    """Class to create EDA visualizations"""
    
    def __init__(self, analyzer: EDAAnalyzer, text_analyzer: TextAnalyzer):
        self.analyzer = analyzer
        self.text_analyzer = text_analyzer
        self.df = analyzer.df
    
    def plot_descriptive_stats(self):
        """Create descriptive statistics plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Headline length distribution
        axes[0,0].hist(self.df['headline_length'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Distribution of Headline Lengths')
        axes[0,0].set_xlabel('Headline Length (characters)')
        axes[0,0].set_ylabel('Frequency')
        
        # Top publishers
        top_publishers = self.df['publisher'].value_counts().head(10)
        axes[0,1].bar(range(len(top_publishers)), top_publishers.values)
        axes[0,1].set_title('Top 10 Publishers by Article Count')
        axes[0,1].set_xlabel('Publisher')
        axes[0,1].set_ylabel('Number of Articles')
        axes[0,1].set_xticks(range(len(top_publishers)))
        axes[0,1].set_xticklabels(top_publishers.index, rotation=45, ha='right')
        
        # Articles by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = self.df['day_of_week'].value_counts().reindex(day_order)
        axes[1,0].bar(day_order, day_counts.values)
        axes[1,0].set_title('Articles Published by Day of Week')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Number of Articles')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Articles by hour
        hour_counts = self.df['hour'].value_counts().sort_index()
        axes[1,1].plot(hour_counts.index, hour_counts.values, marker='o')
        axes[1,1].set_title('Articles Published by Hour of Day')
        axes[1,1].set_xlabel('Hour of Day')
        axes[1,1].set_ylabel('Number of Articles')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('notebooks/descriptive_stats.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_time_series_analysis(self):
        """Create time series analysis plots"""
        time_data = self.analyzer.time_series_analysis()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Daily trend
        time_data['daily'].plot(ax=axes[0,0], title='Daily Article Publications', color='blue')
        axes[0,0].set_ylabel('Number of Articles')
        axes[0,0].grid(True, alpha=0.3)
        
        # Weekly trend
        time_data['weekly'].plot(ax=axes[0,1], title='Weekly Article Publications', color='green')
        axes[0,1].set_ylabel('Number of Articles')
        axes[0,1].grid(True, alpha=0.3)
        
        # Day of week distribution
        time_data['day_of_week'].plot(kind='bar', ax=axes[1,0], title='Articles by Day of Week')
        axes[1,0].set_ylabel('Number of Articles')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Hourly distribution
        time_data['hourly'].plot(kind='bar', ax=axes[1,1], title='Articles by Hour of Day')
        axes[1,1].set_ylabel('Number of Articles')
        axes[1,1].set_xlabel('Hour')
        
        plt.tight_layout()
        plt.savefig('notebooks/time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_topic_analysis(self):
        """Create topic modeling visualization"""
        # Extract topics
        lda_model, topics = self.text_analyzer.topic_modeling_lda(self.df['headline_clean'].tolist())
        
        fig, axes = plt.subplots(1, len(topics), figsize=(20, 5))
        if len(topics) == 1:
            axes = [axes]
        
        for idx, topic in enumerate(topics):
            words = [word for word, score in topic[:10]]
            scores = [score for word, score in topic[:10]]
            
            axes[idx].barh(range(len(words)), scores)
            axes[idx].set_yticks(range(len(words)))
            axes[idx].set_yticklabels(words)
            axes[idx].set_title(f'Topic {idx + 1}')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('notebooks/topic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    loader = DataLoader('path_to_your_dataset.csv')  # Update path
    df = loader.preprocess_data()
    
    # Perform EDA
    analyzer = EDAAnalyzer(loader)
    stats = analyzer.descriptive_statistics()
    publisher_stats = analyzer.publisher_analysis()
    
    # Text analysis
    text_analyzer = TextAnalyzer()
    keywords = text_analyzer.extract_keywords(df['headline_clean'].tolist())
    publisher_keywords = text_analyzer.analyze_publisher_keywords(df)
    
    # Create visualizations
    visualizer = EDAVisualizer(analyzer, text_analyzer)
    visualizer.plot_descriptive_stats()
    visualizer.plot_time_series_analysis()
    visualizer.plot_topic_analysis()
    
    # Print key findings
    print("=== DESCRIPTIVE STATISTICS ===")
    print(f"Total Articles: {stats['total_articles']}")
    print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(f"Number of Publishers: {stats['publishers_count']}")
    print(f"Average Headline Length: {stats['headline_length_stats']['mean']:.2f} characters")
    
    print("\n=== TOP PUBLISHERS ===")
    print(publisher_stats.head(10))
    
    print("\n=== TOP KEYWORDS ===")
    for word, score in keywords[:15]:
        print(f"{word}: {score:.4f}")