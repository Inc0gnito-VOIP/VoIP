"""
train.py v2 — RF + XGBoost + Voting 앙상블 학습 (6피처)
  피처: avg_latency, avg_jitter, iat_variance, packet_loss, seq_gap_rate, codec_mismatch
  출력: model.pkl (VotingClassifier), train_result.txt
"""

import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

FEATURES = [
    'avg_latency', 'avg_jitter', 'iat_variance',
    'packet_loss', 'seq_gap_rate', 'codec_mismatch'
]

def main():
    # ── 데이터 로드
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'sessions.csv'
    df = pd.read_csv(csv_path)
    print(f"데이터: {csv_path} ({len(df)}세션)")
    print(f"  정상: {(df['label']==0).sum()} / SIM Box: {(df['label']==1).sum()}")
    print(f"  피처: {FEATURES}\n")

    X = df[FEATURES].values
    y = df['label'].values

    # ── 모델 정의
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='soft'
    )

    # ── 5-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in [('RF', rf), ('XGB', xgb), ('Ensemble', ensemble)]:
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        results.append(f"{name}: F1={scores.mean():.4f} (±{scores.std():.4f})")
        print(f"[CV] {results[-1]}")

    # ── 전체 데이터로 최종 학습
    print("\n전체 데이터로 최종 학습 중...")
    ensemble.fit(X, y)
    y_pred = ensemble.predict(X)
    print("\n[Train] Ensemble Classification Report:")
    print(classification_report(y, y_pred, target_names=['정상', 'SIM Box']))

    # ── Feature Importance
    rf_final = ensemble.named_estimators_['rf']
    xgb_final = ensemble.named_estimators_['xgb']

    print("Feature Importance (RF):")
    for fname, imp in sorted(zip(FEATURES, rf_final.feature_importances_), key=lambda x: -x[1]):
        print(f"  {fname}: {imp:.4f} ({imp*100:.1f}%)")

    print("\nFeature Importance (XGB):")
    for fname, imp in sorted(zip(FEATURES, xgb_final.feature_importances_), key=lambda x: -x[1]):
        print(f"  {fname}: {imp:.4f} ({imp*100:.1f}%)")

    # ── 저장
    with open('model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("\nmodel.pkl 저장 완료")

    # ── 결과 파일
    with open('train_result.txt', 'w') as f:
        f.write(f"데이터: {csv_path} ({len(df)}세션)\n")
        f.write(f"피처: {FEATURES}\n\n")
        f.write("5-Fold CV:\n")
        for r in results:
            f.write(f"  {r}\n")
        f.write(f"\nFeature Importance (RF):\n")
        for fname, imp in sorted(zip(FEATURES, rf_final.feature_importances_), key=lambda x: -x[1]):
            f.write(f"  {fname}: {imp:.4f} ({imp*100:.1f}%)\n")
        f.write(f"\nFeature Importance (XGB):\n")
        for fname, imp in sorted(zip(FEATURES, xgb_final.feature_importances_), key=lambda x: -x[1]):
            f.write(f"  {fname}: {imp:.4f} ({imp*100:.1f}%)\n")
    print("train_result.txt 저장 완료")


if __name__ == '__main__':
    main()
