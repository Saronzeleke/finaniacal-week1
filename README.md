# Financial News & Stock Analysis Project

A comprehensive analysis project exploring the relationship between financial news sentiment and stock price movements.

## 📋 Project Overview

This project implements a three-task analysis pipeline:

- **Task 1**: Exploratory Data Analysis on financial news headlines

- **Task 2**: Quantitative analysis of stock data with technical indicators  

- **Task 3**: Correlation analysis between news sentiment and stock returns

## 🏗️ Project Structure

finaniacal-week1/

├── data/
│ 
│ │ ├── raw_analyst_ratings.csv

│ │ └── NVDA.csv
│
├── notebooks/ # Jupyter notebooks for each task

│ ├── eda.ipynb

│ ├── quantitative_analysis.ipynb

│ └── correlation_analysis.ipynb

├── src/ # Source code modules

│ ├── eda.py

│ ├── technical_analysis.py

│ └── correlation_analysis.py

├── results/ # Analysis results and visualizations

├── tests/ # Unit tests

│ ├── /test_eda.py
      /test_financial_analysis.py

│ └── task3/

├── scripts/ # Utility 

└── requirements.txt # Project dependencies

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+

- Git

### Installation Steps

1. **Clone the repository**

   git clone https://github.com/Saronzeleke/finaniacal-week1.git

   cd finaniacal-week1

2. **Create virtual environment (recommended)**

    python -m venv venv

    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**

   pip install -r requirements.txt

4. **Download NLTK data (for sentiment analysis)**

   python scripts/setup_nltk.py

# 📊 Task Implementations

**Task 1: Exploratory Data Analysis (EDA)**

Status: ✅ COMPLETED

**Implementation:**

GitHub repository creation and task-1 branch setup

Frequent, descriptive commit messages

Comprehensive EDA covering:

Headline length statistics and distribution

Publisher analysis and counts

Publication date trends and time series analysis

Text analysis and topic exploration

**Files:**

notebooks/eda_analysis.ipynb

src/data_loader.py

tests/task1/test_eda.py

**Task 2: Quantitative Analysis**

Status: ✅ COMPLETED

**Implementation:**

Merged Task 1 branch into main

Created task-2 branch

Loaded and prepared stock price data

Computed technical indicators:

Moving Averages (MA)

Relative Strength Index (RSI)

Moving Average Convergence Divergence (MACD)

Generated clear visualizations showing indicator impact on stock prices

**Files:**

notebooks/financial_analysis.ipynb

src/financial_analysis.py

tests/test_technical_analysis.py

**Task 3: Correlation Analysis**

Status: ✅ COMPLETED

**Implementation:**

Merged Task 2 branch into main

Created task-3 branch

Normalized dates between news and stock data

Performed sentiment analysis on news headlines using TextBlob

Computed daily returns from stock prices

Calculated Pearson correlation coefficient between sentiment and returns

Generated statistical significance testing and visualizations

**Files:**

notebooks/correlation_analysis.ipynb

src/correlation_analysis.py

tests/task3/test_correlation_criteria.py

# 📈 Key Findings

**Task 1 EDA Results**

Analyzed financial news headline patterns and distributions

Identified key publishers and publication trends

Explored temporal patterns in financial news coverage

**Task 2 Technical Analysis**

Successfully computed key technical indicators

Visualized relationships between indicators and price movements

Demonstrated practical application of quantitative analysis

**Task 3 Correlation Analysis**

Pearson Correlation: [Result will be shown after execution]

Statistical Significance: [Result will be shown after execution]

Relationship strength between news sentiment and stock returns

# 🛠️ Usage

Running Individual Tasks

**Task 1: EDA**

python src/data_loader.py

# or

jupyter notebook notebooks/eda_analysis.ipynb

**Task 2: Quantitative Analysis**

python src/financial_analysis.py

# or

jupyter notebook notebooks/financial_analysis.ipynb

**Task 3: Correlation Analysis**

python src/correlation_analysis.py

# or

jupyter notebook notebooks/correlation_analysis.ipynb

Running Tests

# Run all tests

python -m pytest tests/

# Run specific task tests

python -m pytest tests/text_eda.py

python -m pytest tests/test_financial_analysis.py

python -m pytest tests/task3/

# 📁 Data Sources

**News Data: Financial news headlines with publication dates and publishers**

**Stock Data: NVDA (NVIDIA Corporation) historical price data**

# 🔧 Dependencies

**Key packages used:**

pandas - Data manipulation and analysis

numpy - Numerical computations

matplotlib & seaborn - Data visualization

textblob - Sentiment analysis

scipy - Statistical analysis

ta-lib - Technical indicators

jupyter - Interactive notebooks

See requirements.txt for complete list.

# 👥 Git & GitHub Practices

Branch Strategy

main - Production-ready code

task-1 - Exploratory Data Analysis implementation

task-2 - Quantitative Analysis implementation

task-3 - Correlation Analysis implementation

# Commit Standards

Frequent, descriptive commit messages

Feature-based commits with clear purposes

Proper branch management and pull requests

# 📝 License

This project is for educational purposes as part of a financial analysis assignment.

# 🤝 Contributing

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📞 Contact
For questions about this project, please open an issue on GitHub.

Last Updated: 2025
