"""
Technical Analysis with TA-Lib - Required Implementation
"""
import talib
import pandas as pd
import matplotlib.pyplot as plt

def calculate_technical_indicators(df):
    """
    Calculate technical indicators using TA-Lib as required
    """
    # Moving Averages
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    
    # RSI
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
    
    # Additional indicators to show proficiency
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(df['Close'])
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    
    return df

def create_technical_visualizations(df):
    """
    Create required technical indicator visualizations
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Price with Moving Averages
    ax1.plot(df['Date'], df['Close'], label='Close', linewidth=1)
    ax1.plot(df['Date'], df['SMA_20'], label='20-day SMA', linewidth=1)
    ax1.plot(df['Date'], df['SMA_50'], label='50-day SMA', linewidth=1)
    ax1.set_title('TA-Lib: Moving Averages')
    ax1.legend()
    
    # RSI
    ax2.plot(df['Date'], df['RSI_14'], label='RSI-14', color='orange')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='green', alpha=0.7)
    ax2.set_title('TA-Lib: RSI Indicator')
    
    # MACD
    ax3.plot(df['Date'], df['MACD'], label='MACD', color='blue')
    ax3.plot(df['Date'], df['MACD_Signal'], label='Signal', color='red')
    ax3.set_title('TA-Lib: MACD')
    ax3.legend()
    
    # Bollinger Bands
    ax4.plot(df['Date'], df['Close'], label='Close', color='black')
    ax4.plot(df['Date'], df['BB_Upper'], label='Upper Band', linestyle='--')
    ax4.plot(df['Date'], df['BB_Lower'], label='Lower Band', linestyle='--')
    ax4.set_title('TA-Lib: Bollinger Bands')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(r'C:/Users/admin/finaniacal-week1/results/ta_lib_indicators.png', dpi=300, bbox_inches='tight')
    plt.show()

# Demonstrate usage
if __name__ == "__main__":
    df = pd.read_csv(r'C:\Users\admin\finaniacal-week1\data\NVDA.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = calculate_technical_indicators(df)
    create_technical_visualizations(df)
    
    print("TA-Lib Implementation Completed Successfully")
    print(f"Indicators calculated: SMA, RSI, MACD, Bollinger Bands, Stochastic")