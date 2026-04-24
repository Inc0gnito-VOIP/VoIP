"""
eval_test.py v2 — 테스트 데이터 평가 (6피처)
  model.pkl 로드 → test_sessions.csv 평가
  RF / XGB / Ensemble 각각 F1, Precision, Recall
  Premium 구간(latency < 25ms) 별도 분석
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score

FEATURES = [
    'avg_latency', 'avg_jitter', 'iat_variance',
    'packet_loss', 'seq_gap_rate', 'codec_mismatch'
]

def main():
    # 모델 로드
    with open('model.pkl', 'rb') as f:
        ensemble = pickle.load(f)

    rf_model  = ensemble.named_estimators_['rf']
    xgb_model = ensemble.named_estimators_['xgb']

    # 테스트 데이터 로드
    df = pd.read_csv('test_sessions.csv')
    print(f"테스트 데이터: {len(df)}세션")
    print(f"  정상: {(df['label']==0).sum()} / SIM Box: {(df['label']==1).sum()}\n")

    X = df[FEATURES].values
    y = df['label'].values

    # 각 모델 평가
    for name, model in [('RF', rf_model), ('XGB', xgb_model), ('Ensemble', ensemble)]:
        y_pred = model.predict(X)
        f1 = f1_score(y, y_pred)
        print(f"{'='*50}")
        print(f"[{name}] F1={f1:.4f}")
        print(classification_report(y, y_pred, target_names=['정상', 'SIM Box']))

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"  TP={tp} FP={fp} FN={fn} TN={tn} FPR={fpr:.4f}")
        print()

    # Premium 구간 분석 (SIM Box 중 latency < 25ms)
    simbox = df[df['label'] == 1]
    premium_mask = simbox['avg_latency'] < 0.025
    premium = simbox[premium_mask]

    if len(premium) > 0:
        print(f"{'='*50}")
        print(f"[Premium 구간 분석] latency < 25ms: {len(premium)}세션 / 전체 SIM Box {len(simbox)}세션")
        X_prem = premium[FEATURES].values
        y_prem = premium['label'].values

        for name, model in [('RF', rf_model), ('XGB', xgb_model), ('Ensemble', ensemble)]:
            y_pred = model.predict(X_prem)
            detected = y_pred.sum()
            print(f"  {name}: {detected}/{len(premium)} 탐지 ({detected/len(premium)*100:.1f}%)")
    else:
        print("[Premium 구간] 해당 세션 없음")

    # Stealth 구간 분석 (SIM Box 중 15ms < latency < 30ms, loss=0, codec=0)
    stealth_mask = (
        (simbox['avg_latency'] > 0.015) &
        (simbox['avg_latency'] < 0.030) &
        (simbox['packet_loss'] == 0) &
        (simbox['codec_mismatch'] == 0)
    )
    stealth = simbox[stealth_mask]

    if len(stealth) > 0:
        print(f"\n[Stealth 구간 분석] 15~30ms + loss=0 + codec=0: {len(stealth)}세션")
        X_st = stealth[FEATURES].values

        for name, model in [('RF', rf_model), ('XGB', xgb_model), ('Ensemble', ensemble)]:
            y_pred = model.predict(X_st)
            detected = y_pred.sum()
            print(f"  {name}: {detected}/{len(stealth)} 탐지 ({detected/len(stealth)*100:.1f}%)")
    else:
        print("\n[Stealth 구간] 해당 세션 없음")


if __name__ == '__main__':
    main()
