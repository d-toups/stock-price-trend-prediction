# Stock Price Trend Prediction Model – Python

Machine Learning project that classifies whether a stock is trending upward using technical features and two classifiers (Decision Tree and Naïve Bayes).

## Project Overview
Built and compared **Naïve Bayes** and **Decision Tree** classifiers to predict if a stock will trend upward based on historical daily price data. Focused on feature engineering and model evaluation across different time windows and price movement thresholds.

## Key Features & Methodology
- Engineered two technical features: **skewness of returns** and **relative trading volume**
- Tested multiple evaluation windows (14-day and 60-day) and price movement thresholds (5% and 10%)
- Compared model performance using accuracy, precision, recall, and F1-score

## Results
- **Naïve Bayes** outperformed the Decision Tree in most configurations
- Best performance: **70% accuracy** (Naïve Bayes) on 60-day window with 5% price movement threshold
- Demonstrated the difficulty and limitations of short-term stock price prediction

## Repository Structure

```text
stock-price-trend-prediction/
├── Stock Analysis.py
├── FSM_daily.csv
├── RIG_daily.csv
├── requirements.txt
├── README.md
└── LICENSE
```
## Technologies
- Python
- pandas
- scikit-learn (DecisionTreeClassifier, GaussianNB)
- mtplotlib / seaborn

## How to Run
```bash
pip install -r requirements.txt
python "Stock Ahalysis.py"
```

## Key Learnings
- Stock price prediction is extremely challenging due to market efficiency
- Feature engineering and proper time-series validation are critical
- Naive BAyes can sometimes outperform more complex models on small, noisy financial datasets
