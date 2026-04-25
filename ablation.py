"""
ablation.py — 피처 하나씩 제거하며 F1 변화 측정 (Ceiling Effect 대응)
사용법: python ablation.py data/processed/sessions.csv
"""
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

FEATURES = ['avg_latency', 'avg_jitter', 'iat_variance', 'packet_loss',
            'seq_gap_rate', 'codec_mismatch', 'call_pattern', 'concurrent_calls']

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/sessions.csv'
df = pd.read_csv(csv_path)
X = df[FEATURES]
y = df['label']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)

print("=" * 75)
print("  Ablation Study — 피처별 제거 시 F1 변화")
print("=" * 75)
print(f"  데이터: {len(df)}세션 (정상 {(y==0).sum()} / SIM Box {(y==1).sum()})")
print(f"  피처: {len(FEATURES)}개")
print("=" * 75)
print(f"{'제거 피처':<22} {'RF CV F1':>10} {'XGB CV F1':>10} {'RF 하락':>10} {'XGB 하락':>10}")
print("-" * 75)

# Baseline (전체 피처)
rf_base  = cross_val_score(rf,  X, y, cv=cv, scoring='f1').mean()
xgb_base = cross_val_score(xgb, X, y, cv=cv, scoring='f1').mean()
print(f"{'(없음 — Baseline)':<22} {rf_base:>10.6f} {xgb_base:>10.6f} {'—':>10} {'—':>10}")
print("-" * 75)

# 피처별 ablation
results = []
for feat in FEATURES:
    remaining = [f for f in FEATURES if f != feat]
    X_abl = df[remaining]

    rf_f1  = cross_val_score(rf,  X_abl, y, cv=cv, scoring='f1').mean()
    xgb_f1 = cross_val_score(xgb, X_abl, y, cv=cv, scoring='f1').mean()

    rf_drop  = rf_base  - rf_f1
    xgb_drop = xgb_base - xgb_f1

    results.append((feat, rf_f1, xgb_f1, rf_drop, xgb_drop))
    print(f"{feat:<22} {rf_f1:>10.6f} {xgb_f1:>10.6f} {rf_drop:>+10.6f} {xgb_drop:>+10.6f}")

print("=" * 75)

# 물리 피처 전부 제거
phys = ['avg_latency', 'avg_jitter', 'iat_variance']
behav = [f for f in FEATURES if f not in phys]
X_no_phys = df[behav]
rf_np  = cross_val_score(rf,  X_no_phys, y, cv=cv, scoring='f1').mean()
xgb_np = cross_val_score(xgb, X_no_phys, y, cv=cv, scoring='f1').mean()
print(f"{'물리 3개 전부 제거':<22} {rf_np:>10.6f} {xgb_np:>10.6f} {rf_base-rf_np:>+10.6f} {xgb_base-xgb_np:>+10.6f}")

# 행동 피처 전부 제거
X_no_behav = df[phys]
rf_nb  = cross_val_score(rf,  X_no_behav, y, cv=cv, scoring='f1').mean()
xgb_nb = cross_val_score(xgb, X_no_behav, y, cv=cv, scoring='f1').mean()
print(f"{'행동 5개 전부 제거':<22} {rf_nb:>10.6f} {xgb_nb:>10.6f} {rf_base-rf_nb:>+10.6f} {xgb_base-xgb_nb:>+10.6f}")

print("=" * 75)
print("\n결론:")
top_rf  = max(results, key=lambda x: x[3])
top_xgb = max(results, key=lambda x: x[4])
print(f"  RF  가장 큰 하락: {top_rf[0]} 제거 시 -{top_rf[3]:.6f}")
print(f"  XGB 가장 큰 하락: {top_xgb[0]} 제거 시 -{top_xgb[4]:.6f}")
