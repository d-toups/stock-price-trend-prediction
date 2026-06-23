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
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries imported successfully")

# Create data directory
os.makedirs('/content/data', exist_ok=True)

# Define file paths
fsm_path = '/content/data/FSM_daily.pkl'
rig_path = '/content/data/RIG_daily.pkl'

# ====================== DOWNLOAD IF MISSING ======================
print("📥 Checking for data files...")

if not os.path.exists(fsm_path) or not os.path.exists(rig_path):
    print("   → Pickle files not found. Downloading fresh data from Yahoo Finance...\n")
    
    # Install yfinance if not present
    try:
        import yfinance as yf
    except ImportError:
        print("   Installing yfinance...")
        #!pip install yfinance -q
        #import yfinance as yf
    
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

# ====================== LOAD DATA ======================
fsm = pd.read_pickle(fsm_path)
rig = pd.read_pickle(rig_path)

print("\n✅ Data successfully loaded!")
print(f"FSM shape: {fsm.shape} | Date range: {fsm.index[0].date()} to {fsm.index[-1].date()}")
print(f"RIG shape: {rig.shape} | Date range: {rig.index[0].date()} to {rig.index[-1].date()}")

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

def evaluate_models(X, y, stock_name, window, threshold, verbose=True):
    """
    Train Random Forest + Naive Bayes.
    Set verbose=False to suppress detailed output during full runs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # === Random Forest ===
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    # === Naive Bayes ===
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    
    # === Only print when verbose=True ===
    if verbose:
        print(f"\n=== {stock_name} | Window={window} | Threshold={threshold} ===")
        print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
        print(classification_report(y_test, y_pred_rf, 
                                    target_names=["Not Trending", "Trending"], 
                                    zero_division=0))
        
        print(f"Naive Bayes Accuracy:   {accuracy_score(y_test, y_pred_nb):.4f}")
        print(classification_report(y_test, y_pred_nb, 
                                    target_names=["Not Trending", "Trending"], 
                                    zero_division=0))
    
    return rf, nb, y_test, y_pred_rf, y_pred_nb

summary = []

print("🚀 Starting full analysis across all combinations...\n")

for symbol, df_raw in zip(['FSM', 'RIG'], [fsm, rig]):
    for window in WINDOWS:
        for thresh in THRESHOLDS:
            features = engineer_features(df_raw, window, thresh)
            X = features[['skewness', 'relative_volume']].values
            y = features['target'].values
            
            # Run with verbose=False to keep output clean
            rf_model, nb_model, y_test, y_pred_rf, y_pred_nb = evaluate_models(
                X, y, symbol, window, thresh, verbose=False
            )
            
            # Store results
            summary.append({
                'Stock': symbol,
                'Window': window,
                'Threshold': thresh,
                'Threshold_%': f"{int((thresh-1)*100)}%",
                'Model': 'Random Forest',
                'Accuracy': accuracy_score(y_test, y_pred_rf)
            })
            summary.append({
                'Stock': symbol,
                'Window': window,
                'Threshold': thresh,
                'Threshold_%': f"{int((thresh-1)*100)}%",
                'Model': 'Naive Bayes',
                'Accuracy': accuracy_score(y_test, y_pred_nb)
            })

print("✅ Full analysis completed!\n")

report_dir = 'reports'
combined_report_path = os.path.join(report_dir, "FULL_MODEL_REPORTS.txt")

summary = []

print("🚀 Running full evaluation...\n")

with open(combined_report_path, 'w') as f:
    f.write("STOCK TREND PREDICTION - FULL MODEL REPORTS\n")
    f.write("="*80 + "\n\n")
    
    for symbol, df_raw in zip(['FSM', 'RIG'], [fsm, rig]):
        for window in WINDOWS:
            for thresh in THRESHOLDS:
                features = engineer_features(df_raw, window, thresh)
                X = features[['skewness', 'relative_volume']].values
                y = features['target'].values
                
                # Run models silently (no console spam)
                rf_model, nb_model, y_test, y_pred_rf, y_pred_nb = evaluate_models(
                    X, y, symbol, window, thresh, verbose=False
                )
                
                # === Write detailed reports to file only ===
                f.write(f"STOCK: {symbol} | WINDOW: {window} days | "
                        f"THRESHOLD: {thresh} ({int((thresh-1)*100)}%)\n")
                f.write("-"*70 + "\n")
                f.write(f"Samples: {len(y)}\n\n")
                
                f.write("RANDOM FOREST\n")
                f.write(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")
                f.write(classification_report(y_test, y_pred_rf,
                                            target_names=["Not Trending", "Trending"], 
                                            zero_division=0))
                
                f.write("\nNAIVE BAYES\n")
                f.write(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\n")
                f.write(classification_report(y_test, y_pred_nb,
                                            target_names=["Not Trending", "Trending"], 
                                            zero_division=0))
                f.write("\n" + "="*80 + "\n\n")
                
                # Store summary data
                summary.append({
                    'Stock': symbol,
                    'Window': window,
                    'Threshold': thresh,
                    'Threshold_%': f"{int((thresh-1)*100)}%",
                    'Model': 'Random Forest',
                    'Accuracy': accuracy_score(y_test, y_pred_rf)
                })
                summary.append({
                    'Stock': symbol,
                    'Window': window,
                    'Threshold': thresh,
                    'Threshold_%': f"{int((thresh-1)*100)}%",
                    'Model': 'Naive Bayes',
                    'Accuracy': accuracy_score(y_test, y_pred_nb)
                })

print(f"✅ Detailed reports saved to: {combined_report_path}")
print("   (You can download it from the file browser)\n")

# ====================== SUMMARY TABLE (in notebook) ======================
summary_df = pd.DataFrame(summary)

# Nice ordering
summary_df['order'] = summary_df.apply(
    lambda x: (x['Threshold'], x['Window'], 
               0 if x['Stock']=='FSM' else 1,
               0 if x['Model']=='Random Forest' else 1), axis=1
)
summary_df = summary_df.sort_values('order').drop(columns='order')

print("=== SUMMARY TABLE ===")
display_cols = ['Stock', 'Window', 'Threshold_%', 'Model', 'Accuracy']
display(summary_df[display_cols].round(4).style.set_caption("Model Performance Summary"))

if 'summary_df' not in globals():
    print("⚠️ Please run the analysis cell first!")
else:
    for stock in ['FSM', 'RIG']:
        for thresh in sorted(THRESHOLDS):
            plot_data = summary_df[
                (summary_df['Stock'] == stock) & 
                (summary_df['Threshold'] == thresh)
            ].copy()
            
            plt.figure(figsize=(10, 6))
            
            sns.barplot(
                data=plot_data,
                x='Window',
                y='Accuracy',
                hue='Model',
                palette=['#1f77b4', '#ff7f0e'],
                dodge=True,
                errorbar=None
            )
            
            plt.title(f'{stock} - Threshold = {thresh} ({int((thresh-1)*100)}% above avg)\n'
                      f'Random Forest vs Naive Bayes', fontsize=14, pad=20)
            
            plt.ylabel('Accuracy', fontsize=12)
            plt.xlabel('Rolling Window Size (Days)', fontsize=12)
            plt.ylim(0.45, 1.0)
            plt.legend(title='Model', loc='lower right')
            
            # Value labels
            ax = plt.gca()
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=4)
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
