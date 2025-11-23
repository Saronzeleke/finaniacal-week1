# Financial News Analysis and Quantitative Finance Platform

## Project Overview

A comprehensive data science project combining financial news analysis with quantitative technical analysis using 

Python, TA-Lib, and machine learning techniques.

## 📁 Project Structure

├── .vscode/

│ └── settings.json

├── .github/

│ └── workflows/

│ └── unittests.yml

├── .gitignore

├── requirements.txt

├── README.md

├── src/

│ ├── init.py

│ ├── data_loader.py

│ ├── text_analyzer.py

│ └── financial_analyzer.py

├── notebooks/

│ ├── eda_analysis.ipynb

│ ├── financial_analysis.ipynb

│ └── images/

├── tests/

│ ├── init.py

│ ├── test_eda.py

│ └── test_financial_analysis.py

└── scripts/

├── init.py

└── README.md


## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+

- Git

### 1. Clone Repository

git clone (https://github.com/Saronzeleke/finaniacal-week1.git)

cd finaniacal-week1

# 2. Create Virtual Environment

python -m venv my_env

source my_env/bin/activate  # Linux/Mac

# OR

my_env\Scripts\activate    # Windows

# 3. Install Dependencies

pip install -r requirements.txt

# 4. Setup TA-Lib (Required for Technical Analysis)

Windows:

#Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl

Linux/Mac:

# Install TA-Lib library first

brew install ta-lib  # Mac

# OR

sudo apt-get install ta-lib  # Linux

pip install TA-Lib

# 📊 Task 1: Exploratory Data Analysis (EDA)

# Features Implemented

Descriptive Statistics: Headline length analysis, publisher activity

Text Analysis: Topic modeling, keyword extraction using LDA

Time Series Analysis: Publication frequency trends, temporal patterns

Publisher Analysis: Domain extraction, activity metrics

# Run EDA Analysis

jupyter notebook notebooks/eda_analysis.ipynb

# Key Outputs

Descriptive statistics and visualizations

Topic modeling results

Publisher activity charts

Time series trends

# 📈 Task 2: Quantitative Financial Analysis

# Features Implemented

Technical Indicators: RSI, MACD, Moving Averages, Bollinger Bands

Financial Metrics: Sharpe ratio, volatility, cumulative returns

Visualization: Comprehensive charting of indicators

Comparative Analysis: Multi-stock performance comparison

# Run Financial Analysis

jupyter notebook notebooks/financial_analysis.ipynb

# Key Outputs

Technical indicator charts

Financial metrics reports

Support/resistance levels

Risk-return analysis

# 🧪 Testing

Run the test suite:

python -m pytest tests/ -v 

# 🔄 Git Workflow

Branch Strategy

main: Production-ready code

task-1: EDA analysis implementation

task-2: Financial analysis implementation

Commit Convention

feat: New features

fix: Bug fixes

docs: Documentation

test: Test cases

merge: Branch integrations

# 📈 Key Performance Indicators (KPIs)

# Task 1 KPIs

✅ Dev Environment Setup

✅ EDA Analysis Completeness

✅ Text Analysis Accuracy

✅ Repository Organization

# Task 2 KPIs

✅ Technical Indicator Accuracy

✅ Data Analysis Completeness

✅ Self-learning Demonstration

✅ Visualization Quality

# 🛠 Technical Stack

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

NLP: NLTK, Scikit-learn, Gensim

Technical Analysis: TA-Lib, PyNance

Testing: Pytest

CI/CD: GitHub Actions

# 📝 Usage Examples

Load and Analyze News Data

from src.data_loader import DataLoader

from src.text_analyzer import TextAnalyzer

loader = DataLoader(r'C:\Users\admin\finaniacal-week1\data\raw_analyst_ratings.csv')

df = loader.preprocess_data()

analyzer = EDAAnalyzer(loader)

stats = analyzer.descriptive_statistics()

# Financial Analysis 

from src.financial_analyzer import FinancialDataLoader, TechnicalAnalyzer

loader = FinancialDataLoader()

data = loader.load_stock_data('AAPL')

tech = TechnicalAnalyzer()

data_with_indicators = tech.calculate_all_indicators(data)

# 🤝 Contributing

Fork the repository

Create feature branch (git checkout -b feature/AmazingFeature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/AmazingFeature)

Open Pull Request

# 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

# 👥 Authors

Your Name - Saron Zeleke

# 🙏 Acknowledgments

TA-Lib community for technical analysis functions

Yahoo Finance for financial data

Scikit-learn for machine learning utilities 
