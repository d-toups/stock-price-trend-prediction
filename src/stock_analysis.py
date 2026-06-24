'''
Author: Dennis Toups
Date: 24 June 2026
'''

# ======================== IMPORTS AND SETUP ==================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from joblib import parallel_backend
import os
import yfinance as yf

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
     accuracy_score, 
     classification_report, 
     confusion_matrix,
     ConfusionMatrixDisplay
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries imported successfully")

## Runtime Information
import sys
import platform
import sklearn
import multiprocessing
import psutil

print("🔧 Runtime Information")
print("=" * 55)
print(f"Python version     : {sys.version.split()[0]}")
print(f"Platform           : {platform.platform()}")
print(f"Pandas version     : {pd.__version__}")
print(f"NumPy version      : {np.__version__}")
print(f"Scikit-learn       : {sklearn.__version__}")
print(f"CPU cores available: {multiprocessing.cpu_count()}")
print(f"Available RAM      : {psutil.virtual_memory().total / (1024**3):.1f} GB")
print(f"Current working dir: {os.getcwd()}")

print("\n✅ Runtime ready for reproducible analysis")

# ======================= DATA DOWNLOAD & LOADING =============================
# Create data directory
os.makedirs('/content/data', exist_ok=True)

# Define file paths
fsm_path = '/content/data/FSM_daily.pkl'
rig_path = '/content/data/RIG_daily.pkl'

# Download missing files
print("📥 Checking for data files...")

if not os.path.exists(fsm_path) or not os.path.exists(rig_path):
    print("   → Pickle files not found. Downloading fresh data from Yahoo Finance...\n")
    
    # Install yfinance if not present
    try:
        import yfinance as yf
    except ImportError:
        print("   Installing yfinance...")
        !pip install yfinance -q
        import yfinance as yf
    
    # Download data
    for ticker, filepath in zip(["FSM", "RIG"], [fsm_path, rig_path]):
        print(f"   Downloading {ticker}...")
        df = yf.download(
            ticker, 
            start="2020-11-25", 
            end="2025-11-25", 
            auto_adjust=True,
            progress=False
        )
        df.to_pickle(filepath)
        print(f"   ✅ {ticker} saved ({len(df)} rows)")
    
else:
    print("   ✅ Pickle files already exist. Loading from disk.")

# =============================== LOAD DATA ===================================
fsm = pd.read_pickle(fsm_path)
rig = pd.read_pickle(rig_path)

print("\n✅ Data successfully loaded!")
print(f"FSM shape: {fsm.shape} | Date range: {fsm.index[0].date()} to {fsm.index[-1].date()}")
print(f"RIG shape: {rig.shape} | Date range: {rig.index[0].date()} to {rig.index[-1].date()}")

# =========================== FEATURE ENGINEERING =============================
def engineer_features(raw: pd.DataFrame, window: int, threshold: float):
    """
    Engineer skewness + relative volume features with binary target.
    """
    # Basic validation
    if raw.isna().any().any():
        raise ValueError("Missing or invalid data detected")
    
    data = []
    
    for i in range(window - 1, len(raw)):
        block = raw.iloc[i - window + 1 : i + 1]
        
        # Force clean scalars
        avg_price = float(block['Close'].mean().item())
        last_price = float(block['Close'].iloc[-1].item())
        
        # Normalized prices → skewness
        norm_prices = block['Close'] / avg_price
        skewness = float(norm_prices.skew().item())
        
        # Relative volume
        avg_vol = float(block['Volume'].mean())
        last_vol = float(block['Volume'].iloc[-1])
        rel_volume = last_vol / avg_vol if avg_vol != 0 else 0.0
        
        # Target
        trend = 1 if last_price > (threshold * avg_price) else 0
        
        data.append([skewness, rel_volume, trend])
    
    df_features = pd.DataFrame(data, columns=['skewness', 'relative_volume', 
                                              'target'])
    return df_features

# ====================== MODEL TRAINING AND EVALUATION ========================
def evaluate_models_cv(X, y, stock_name, window, threshold, n_splits=5, verbose=True):
    """
    Time-series aware evaluation using expanding window CV.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)  # or test_size= fixed days if you prefer
    
    rf_scores = []
    nb_scores = []
    fold_reports = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=5,
            min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred_rf)
        rf_scores.append(rf_acc)
        
        # Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        nb_acc = accuracy_score(y_test, y_pred_nb)
        nb_scores.append(nb_acc)
        
        if verbose:
            fold_reports.append(\
            f"Fold {fold+1} | RF: {rf_acc:.4f} | NB: {nb_acc:.4f} | Test size: {len(test_idx)}")
    
    mean_rf = np.mean(rf_scores)
    mean_nb = np.mean(nb_scores)
    
    if verbose:
        print(f"\n=== {stock_name} | Window={window} | Threshold={threshold} ===")
        print(f"Random Forest  CV Accuracy: {mean_rf:.4f} (±{np.std(rf_scores):.4f})")
        print(f"Naive Bayes    CV Accuracy: {mean_nb:.4f} (±{np.std(nb_scores):.4f})")
        for report in fold_reports:
            print(report)
    
    return mean_rf, mean_nb, rf_scores, nb_scores

# ============================= FULL ANALYSIS =================================
summary = []

print("🚀 Starting full TimeSeries CV analysis...\n")

for symbol, df_raw in zip(['FSM', 'RIG'], [fsm, rig]):
    for window in WINDOWS:
        for thresh in THRESHOLDS:
            features = engineer_features(df_raw, window, thresh)
            X = features[['skewness', 'relative_volume']].values
            y = features['target'].values
            
            rf_acc, nb_acc, _, _ = evaluate_models_cv(
                X, y, symbol, window, thresh, n_splits=5, verbose=False
            )
            
            # Store results (use mean CV accuracy)
            summary.append({'Stock': symbol, 'Window': window, 'Threshold': thresh,
                            'Threshold_%': f"{int((thresh-1)*100)}%", 'Model': 'Random Forest',
                            'Accuracy': rf_acc, 'CV': True})
            summary.append({'Stock': symbol, 'Window': window, 'Threshold': thresh,
                            'Threshold_%': f"{int((thresh-1)*100)}%", 'Model': 'Naive Bayes',
                            'Accuracy': nb_acc, 'CV': True})

print("✅ Full analysis completed!\n")

# ====================== RESULTS SUMMARY AND REPORT FILE ======================
# Ensure reports folder exists
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "figures").mkdir(exist_ok=True)

summary = []

print("🚀 Running TimeSeriesSplit CV Analysis across all configurations...\n")

for symbol, df_raw in zip(['FSM', 'RIG'], [fsm, rig]):
    for window in WINDOWS:
        for thresh in THRESHOLDS:
            features = engineer_features(df_raw, window, thresh)
            X = features[['skewness', 'relative_volume']].values
            y = features['target'].values
            
            mean_rf, mean_nb, rf_scores, nb_scores = evaluate_models_cv(
                X, y, symbol, window, thresh, n_splits=5, verbose=False
            )
            
            summary.append({
                'Stock': symbol,
                'Window': window,
                'Threshold': thresh,
                'Threshold_%': f"{int((thresh-1)*100)}%",
                'Model': 'Random Forest',
                'Mean_Accuracy': round(mean_rf, 4),
                'Std_Dev': round(np.std(rf_scores), 4)
            })
            summary.append({
                'Stock': symbol,
                'Window': window,
                'Threshold': thresh,
                'Threshold_%': f"{int((thresh-1)*100)}%",
                'Model': 'Naive Bayes',
                'Mean_Accuracy': round(mean_nb, 4),
                'Std_Dev': round(np.std(nb_scores), 4)
            })

# Summary Dataframe
summary_df = pd.DataFrame(summary)

# Order for easy comparison
summary_df = summary_df.sort_values(
    by=['Threshold', 'Window', 'Stock', 'Model']
).reset_index(drop=True)

print("✅ Analysis Complete!\n")

# Display full summary table
styled_summary = summary_df.style\
    .format({
        'Mean_Accuracy': '{:.4f}',
        'Std_Dev': '{:.4f}'
    })\
    .background_gradient(subset=['Mean_Accuracy'], cmap='RdYlGn', vmin=0.5, vmax=0.85)\
    .set_properties(**{'text-align': 'center'})\
    .set_caption("📊 Model Performance Summary - TimeSeriesSplit CV")\
    .set_table_styles([{
        'selector': 'caption',
        'props': [('font-size', '16px'), ('font-weight', 'bold'), ('text-align', 'center')]
    }])

display(styled_summary)

# Save summary
summary_df.to_csv(REPORT_DIR / "model_summary_timeseries_cv.csv", index=False)
print(f"\n📊 Summary saved to: reports/model_summary_timeseries_cv.csv")

# =============================== VISUALIZATIONS ==============================
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

for stock in ['FSM', 'RIG']:
    for thresh in sorted(THRESHOLDS):
        plot_data = summary_df[
            (summary_df['Stock'] == stock) & 
            (summary_df['Threshold'] == thresh)
        ].copy()
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=plot_data, 
            x='Window', 
            y='Mean_Accuracy', 
            hue='Model',
            palette=['#1f77b4', '#ff7f0e'],
            dodge=True
        )
        
        plt.title(f'{stock} - Threshold = {thresh} ({int((thresh-1)*100)}% above avg)\n'
                  f'Random Forest vs Naive Bayes (TimeSeries CV)', fontsize=14, pad=20)
        plt.ylabel('Mean Accuracy (5-fold CV)', fontsize=12)
        plt.xlabel('Rolling Window Size (Days)', fontsize=12)
        plt.ylim(0.45, 1.0)
        plt.legend(title='Model', loc='lower right')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=4, fontsize=11)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f"{stock}_threshold_{int((thresh-1)*100)}.png"
        plt.savefig(REPORT_DIR / "figures" / filename, dpi=200, bbox_inches='tight')
        plt.show()

print("✅ All visualizations saved to reports/figures/")
