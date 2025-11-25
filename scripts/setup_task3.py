"""
Setup script for Task 3 dependencies
"""

import subprocess
import sys

def install_packages():
    """Install required packages for sentiment analysis"""
    packages = [
        'textblob==0.17.1',
        'nltk==3.8.1'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

def download_nltk_data():
    """Download required NLTK data for TextBlob"""
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('brown')
        nltk.download('movie_reviews')
        print("NLTK data downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

if __name__ == "__main__":
    print("Setting up Task 3 dependencies...")
    install_packages()
    download_nltk_data()
    print("Setup completed!")