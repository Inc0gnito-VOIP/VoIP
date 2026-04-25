from flask import Flask, jsonify, request, render_template_string
import csv
import sys
import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH  = BASE_DIR / 'data' / 'processed' / 'sessions.csv'
MODEL_PATH = BASE_DIR / 'src' / 'model' / 'model.pkl'
UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# rtp_parser 임포트
sys.path.insert(0, str(BASE_DIR / 'src' / 'parser'))
from rtp_parser import extract_features

FEATURES = ['avg_latency', 'avg_jitter', 'iat_variance', 'packet_loss', 'seq_gap_rate',
            'codec_mismatch']

# ─── 모델 로드 ───────────────────────────────────────────────
rf_model, xgb_model, ensemble_model = None, None, None

def load_models():
    global rf_model, xgb_model, ensemble_model
    if MODEL_PATH.exists():
        ensemble_model = joblib.load(MODEL_PATH)
        print(f"[OK] 앙상블 모델 로드: {MODEL_PATH.name}")
        # Voting 앙상블에서 개별 모델 추출
        try:
            rf_model = ensemble_model.named_estimators_['rf']
            xgb_model = ensemble_model.named_estimators_['xgb']
            print(f"[OK] RF + XGB 개별 모델 추출 완료")
        except Exception as e:
            print(f"[WARN] 개별 모델 추출 실패: {e}")
    else:
        print(f"[WARN] 모델 없음: {MODEL_PATH}")

load_models()


def predict_all(feat_dict):
    """피처 dict → RF, XGB, 앙상블 예측 결과 반환 (단일 세션용)"""
    x = pd.DataFrame([[feat_dict[f] for f in FEATURES]], columns=FEATURES)
    result = {}
    if rf_model:
        result['rf_label']  = int(rf_model.predict(x)[0])
        result['rf_prob']   = round(float(rf_model.predict_proba(x)[0][1]), 4)
        result['rf_risk']   = 'HIGH' if result['rf_prob'] > 0.7 else ('MED' if result['rf_prob'] > 0.4 else 'LOW')
    if xgb_model:
        result['xgb_label'] = int(xgb_model.predict(x)[0])
        result['xgb_prob']  = round(float(xgb_model.predict_proba(x)[0][1]), 4)
        result['xgb_risk']  = 'HIGH' if result['xgb_prob'] > 0.7 else ('MED' if result['xgb_prob'] > 0.4 else 'LOW')
    if ensemble_model:
        result['ens_label'] = int(ensemble_model.predict(x)[0])
        result['ens_prob']  = round(float(ensemble_model.predict_proba(x)[0][1]), 4)
        result['ens_risk']  = 'HIGH' if result['ens_prob'] > 0.7 else ('MED' if result['ens_prob'] > 0.4 else 'LOW')
    return result


# ─── 데이터 로드 + 예측 캐시 ─────────────────────────────────
_sessions_cache = None

def load_sessions():
    global _sessions_cache
    if _sessions_cache is not None:
        return _sessions_cache

    sessions = []
    if not CSV_PATH.exists():
        print(f"[WARN] CSV 없음: {CSV_PATH}")
        return sessions

    # CSV 전체 읽기
    rows = []
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        return sessions

    # 배치 예측 (10000행 한 번에)
    X = pd.DataFrame([[float(r[f]) for f in FEATURES] for r in rows], columns=FEATURES)

    rf_labels, rf_probs   = None, None
    xgb_labels, xgb_probs = None, None
    ens_labels, ens_probs = None, None

    if rf_model:
        rf_labels = rf_model.predict(X).tolist()
        rf_probs  = rf_model.predict_proba(X)[:, 1].tolist()
    if xgb_model:
        xgb_labels = xgb_model.predict(X).tolist()
        xgb_probs  = xgb_model.predict_proba(X)[:, 1].tolist()
    if ensemble_model:
        ens_labels = ensemble_model.predict(X).tolist()
        ens_probs  = ensemble_model.predict_proba(X)[:, 1].tolist()

    for i, row in enumerate(rows):
        entry = {
            'session_id':       row['session_id'],
            'avg_latency':      float(row['avg_latency']),
            'avg_jitter':       float(row['avg_jitter']),
            'iat_variance':     float(row['iat_variance']),
            'packet_loss':      float(row['packet_loss']),
            'seq_gap_rate':     float(row['seq_gap_rate']),
            'codec_mismatch':   float(row['codec_mismatch']),
            'label':            int(row['label']),
        }
        if rf_labels:
            p = round(rf_probs[i], 4)
            entry.update({'rf_label': rf_labels[i], 'rf_prob': p,
                          'rf_risk': 'HIGH' if p > 0.7 else ('MED' if p > 0.4 else 'LOW')})
        if xgb_labels:
            p = round(xgb_probs[i], 4)
            entry.update({'xgb_label': xgb_labels[i], 'xgb_prob': p,
                          'xgb_risk': 'HIGH' if p > 0.7 else ('MED' if p > 0.4 else 'LOW')})
        if ens_labels:
            p = round(ens_probs[i], 4)
            entry.update({'ens_label': ens_labels[i], 'ens_prob': p,
                          'ens_risk': 'HIGH' if p > 0.7 else ('MED' if p > 0.4 else 'LOW')})
        sessions.append(entry)

    _sessions_cache = sessions
    print(f"[OK] 세션 {len(sessions)}개 캐시 완료 (RF+XGB+앙상블 배치 예측)")
    return sessions


# ─── 라우트 ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/sessions')
def api_sessions():
    return jsonify(load_sessions())


@app.route('/api/stats')
def api_stats():
    sessions = load_sessions()
    if not sessions:
        return jsonify({'total': 0, 'normal_count': 0, 'fraud_count': 0,
                        'normal_avg_jitter': 0, 'fraud_avg_jitter': 0,
                        'normal_avg_latency': 0, 'fraud_avg_latency': 0,
                        'rf_accuracy': 0, 'xgb_accuracy': 0, 'agreement_rate': 0})

    normal = [s for s in sessions if s['label'] == 0]
    fraud  = [s for s in sessions if s['label'] == 1]

    def avg(lst, key):
        return round(sum(s[key] for s in lst) / len(lst), 6) if lst else 0

    rf_correct  = sum(1 for s in sessions if s.get('rf_label')  == s['label'])
    xgb_correct = sum(1 for s in sessions if s.get('xgb_label') == s['label'])
    ens_correct = sum(1 for s in sessions if s.get('ens_label') == s['label'])
    agree       = sum(1 for s in sessions if s.get('rf_label')  == s.get('xgb_label'))

    return jsonify({
        'total': len(sessions),
        'normal_count': len(normal),
        'fraud_count': len(fraud),
        'normal_avg_jitter': avg(normal, 'avg_jitter'),
        'fraud_avg_jitter':  avg(fraud,  'avg_jitter'),
        'normal_avg_latency': avg(normal, 'avg_latency'),
        'fraud_avg_latency':  avg(fraud,  'avg_latency'),
        'rf_accuracy':  round(rf_correct  / len(sessions) * 100, 2),
        'xgb_accuracy': round(xgb_correct / len(sessions) * 100, 2),
        'ens_accuracy': round(ens_correct / len(sessions) * 100, 2),
        'agreement_rate': round(agree / len(sessions) * 100, 2),
    })


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(force=True)
    try:
        feat = {f: float(data[f]) for f in FEATURES}
    except (KeyError, TypeError) as e:
        return jsonify({'error': f'Missing feature: {e}'}), 400
    pred = predict_all(feat)
    return jsonify({'features': feat, **pred})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
    f = request.files['file']
    if not f.filename or not f.filename.lower().endswith('.pcap'):
        return jsonify({'error': '.pcap 파일만 업로드 가능합니다'}), 400

    filename = secure_filename(f.filename)
    save_path = UPLOAD_DIR / filename
    f.save(str(save_path))

    # 파싱 (label=-1 : 업로드 모드, 정답 없음)
    raw_rows = []
    counter = [0]
    extract_features(str(save_path), -1, raw_rows, counter)
    if not raw_rows:
        return jsonify({'error': 'RTP 세션을 추출할 수 없습니다. PCAP에 UDP/RTP 패킷이 있는지 확인하세요.'}), 422

    raw_sessions = raw_rows
    # 배치 예측
    X = pd.DataFrame([[s[feat] for feat in FEATURES] for s in raw_sessions], columns=FEATURES)
    rf_labels, rf_probs   = None, None
    xgb_labels, xgb_probs = None, None
    ens_labels, ens_probs = None, None
    if rf_model:
        rf_labels = rf_model.predict(X).tolist()
        rf_probs  = rf_model.predict_proba(X)[:, 1].tolist()
    if xgb_model:
        xgb_labels = xgb_model.predict(X).tolist()
        xgb_probs  = xgb_model.predict_proba(X)[:, 1].tolist()
    if ensemble_model:
        ens_labels = ensemble_model.predict(X).tolist()
        ens_probs  = ensemble_model.predict_proba(X)[:, 1].tolist()

    results = []
    for i, s in enumerate(raw_sessions):
        entry = {
            'session_id':       s['session_id'],
            'avg_latency':      s['avg_latency'],
            'avg_jitter':       s['avg_jitter'],
            'iat_variance':     s['iat_variance'],
            'packet_loss':      s['packet_loss'],
            'seq_gap_rate':     s['seq_gap_rate'],
            'codec_mismatch':   s.get('codec_mismatch', 0),
        }
        if rf_labels is not None:
            p = round(rf_probs[i], 4)
            entry.update({'rf_label': rf_labels[i], 'rf_prob': p,
                          'rf_risk': 'HIGH' if p > 0.7 else ('MED' if p > 0.4 else 'LOW')})
        if xgb_labels is not None:
            p = round(xgb_probs[i], 4)
            entry.update({'xgb_label': xgb_labels[i], 'xgb_prob': p,
                          'xgb_risk': 'HIGH' if p > 0.7 else ('MED' if p > 0.4 else 'LOW')})
        if ens_labels is not None:
            p = round(ens_probs[i], 4)
            entry.update({'ens_label': ens_labels[i], 'ens_prob': p,
                          'ens_risk': 'HIGH' if p > 0.7 else ('MED' if p > 0.4 else 'LOW')})
        results.append(entry)

    print(f"[OK] 업로드 분석: {filename} → {len(results)}개 세션")
    return jsonify({'filename': filename, 'session_count': len(results), 'sessions': results})


@app.route('/api/analyze/batch', methods=['POST'])
def api_analyze_batch():
    data = request.get_json(force=True)
    sessions = data if isinstance(data, list) else data.get('sessions', [])
    results = []
    for i, s in enumerate(sessions):
        try:
            feat = {f: float(s[f]) for f in FEATURES}
            pred = predict_all(feat)
            results.append({'index': i, 'features': feat, **pred})
        except (KeyError, TypeError) as e:
            results.append({'index': i, 'error': str(e)})
    return jsonify(results)


# ─── 대시보드 HTML ────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SIM Box 탐지 시스템</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
  :root {
    --color-bg: #0f172a;
    --color-card: #1e293b;
    --color-border: #334155;
    --color-fraud: #ef4444;
    --color-normal: #22c55e;
    --color-warning: #f59e0b;
    --color-text: #f1f5f9;
    --color-muted: #94a3b8;
    --accent: #00d4ff;
    --color-xgb: #a78bfa;
    --color-rf: #22c55e;
    --color-ens: #f59e0b;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--color-bg);color:var(--color-text);font-family:'Syne',sans-serif;min-height:100vh;}
  header{padding:24px 32px;border-bottom:1px solid var(--color-border);display:flex;align-items:center;justify-content:space-between;}
  header h1{font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;}
  header h1 span{color:var(--accent);}
  .header-right{display:flex;flex-direction:column;align-items:flex-end;gap:4px;}
  .status-dot{width:8px;height:8px;background:var(--color-normal);border-radius:50%;display:inline-block;margin-right:8px;box-shadow:0 0 8px var(--color-normal);animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
  .status-text{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--color-normal);}
  .last-load{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--color-muted);}
  main{padding:32px;max-width:1500px;margin:0 auto;}

  /* 상단 스탯 카드 */
  .stats-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:14px;margin-bottom:28px;}
  .stat-card{background:var(--color-card);border:1px solid var(--color-border);border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
  .stat-card.total::before{background:var(--accent);}
  .stat-card.normal::before{background:var(--color-normal);}
  .stat-card.fraud::before{background:var(--color-fraud);}
  .stat-card.ratio::before{background:var(--color-warning);}
  .stat-card.rf-acc::before{background:var(--color-rf);}
  .stat-card.xgb-acc::before{background:var(--color-xgb);}
  .stat-icon{font-size:1.2rem;margin-bottom:6px;display:block;}
  .stat-label{font-size:0.65rem;color:var(--color-muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;font-family:'JetBrains Mono',monospace;}
  .stat-value{font-size:1.7rem;font-weight:800;line-height:1;}
  .stat-card.total .stat-value{color:var(--accent);}
  .stat-card.normal .stat-value{color:var(--color-normal);}
  .stat-card.fraud .stat-value{color:var(--color-fraud);}
  .stat-card.ratio .stat-value{color:var(--color-warning);}
  .stat-card.rf-acc .stat-value{color:var(--color-rf);}
  .stat-card.xgb-acc .stat-value{color:var(--color-xgb);}
  .stat-card.ens-acc::before{background:var(--color-ens);}
  .stat-card.ens-acc .stat-value{color:var(--color-ens);}
  .stat-sub{font-size:0.7rem;color:var(--color-muted);margin-top:5px;font-family:'JetBrains Mono',monospace;}

  /* 모델 배지 범례 */
  .model-legend{display:flex;gap:20px;align-items:center;margin-bottom:16px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;}
  .legend-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px;}

  /* 차트 */
  .charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px;}
  .chart-card{background:var(--color-card);border:1px solid var(--color-border);border-radius:12px;padding:24px;}
  .chart-title{font-size:0.78rem;color:var(--color-muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:18px;font-family:'JetBrains Mono',monospace;}
  .chart-title span{color:var(--accent);margin-right:8px;}

  /* 테이블 */
  .table-card{background:var(--color-card);border:1px solid var(--color-border);border-radius:12px;padding:24px;}
  .table-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;}
  .filter-btns{display:flex;gap:8px;}
  .filter-btn{padding:5px 12px;border-radius:6px;border:1px solid var(--color-border);background:transparent;color:var(--color-muted);cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:0.7rem;transition:all 0.2s;}
  .filter-btn.active,.filter-btn:hover{border-color:var(--accent);color:var(--accent);}
  table{width:100%;border-collapse:collapse;}
  th{text-align:left;padding:9px 12px;font-size:0.64rem;color:var(--color-muted);text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid var(--color-border);font-family:'JetBrains Mono',monospace;white-space:nowrap;}
  th.th-rf{color:var(--color-rf);}
  th.th-xgb{color:var(--color-xgb);}
  td{padding:10px 12px;font-size:0.78rem;font-family:'JetBrains Mono',monospace;border-bottom:1px solid rgba(51,65,85,0.5);}
  tr:last-child td{border-bottom:none;}
  tr:hover td{background:rgba(255,255,255,0.02);}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.64rem;font-weight:700;letter-spacing:0.05em;}
  .badge-normal{background:rgba(34,197,94,0.1);color:var(--color-normal);border:1px solid rgba(34,197,94,0.2);}
  .badge-fraud{background:rgba(239,68,68,0.1);color:var(--color-fraud);border:1px solid rgba(239,68,68,0.2);}
  .badge-high{background:rgba(239,68,68,0.12);color:var(--color-fraud);border:1px solid rgba(239,68,68,0.3);}
  .badge-med{background:rgba(245,158,11,0.12);color:var(--color-warning);border:1px solid rgba(245,158,11,0.3);}
  .badge-low{background:rgba(34,197,94,0.12);color:var(--color-normal);border:1px solid rgba(34,197,94,0.3);}
  .badge-disagree{background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid rgba(251,191,36,0.4);}
  .prob-bar{display:flex;align-items:center;gap:6px;}
  .prob-track{width:48px;height:4px;background:var(--color-border);border-radius:2px;overflow:hidden;}
  .prob-fill{height:100%;border-radius:2px;}
  .prob-rf{background:var(--color-rf);}
  .prob-xgb{background:var(--color-xgb);}
  .prob-ens{background:var(--color-ens);}
  .disagree-row td{background:rgba(251,191,36,0.04);}

  /* 업로드 패널 */
  .upload-panel{background:var(--color-card);border:1px solid var(--color-border);border-radius:12px;padding:24px;margin-bottom:28px;}
  .upload-panel-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;}
  .upload-zone{border:2px dashed var(--color-border);border-radius:8px;padding:28px;text-align:center;cursor:pointer;transition:all 0.2s;position:relative;}
  .upload-zone:hover,.upload-zone.dragover{border-color:var(--accent);background:rgba(0,212,255,0.04);}
  .upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
  .upload-zone-icon{font-size:2rem;margin-bottom:8px;}
  .upload-zone-text{font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--color-muted);}
  .upload-zone-text span{color:var(--accent);}
  .upload-status{margin-top:14px;font-family:'JetBrains Mono',monospace;font-size:0.78rem;min-height:20px;}
  .upload-status.parsing{color:var(--color-warning);}
  .upload-status.done{color:var(--color-normal);}
  .upload-status.error{color:var(--color-fraud);}
  .upload-summary{display:flex;gap:20px;margin-top:16px;flex-wrap:wrap;}
  .upload-stat{background:rgba(255,255,255,0.03);border:1px solid var(--color-border);border-radius:8px;padding:12px 18px;font-family:'JetBrains Mono',monospace;}
  .upload-stat-label{font-size:0.64rem;color:var(--color-muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;}
  .upload-stat-value{font-size:1.3rem;font-weight:700;}
  .us-total{color:var(--accent);}
  .us-normal{color:var(--color-normal);}
  .us-fraud{color:var(--color-fraud);}
  .us-agree{color:var(--color-xgb);}
</style>
</head>
<body>
<header>
  <h1>INC0GNITO <span>//</span> SIM Box Detection</h1>
  <div class="header-right">
    <div><span class="status-dot"></span><span class="status-text" id="status-text">서버 연결 중...</span></div>
    <div class="last-load" id="last-load"></div>
  </div>
</header>
<main>
  <!-- ── PCAP 업로드 패널 ── -->
  <div class="upload-panel">
    <div class="upload-panel-header">
      <div class="chart-title" style="margin:0"><span>▸</span>PCAP Upload &amp; Analyze</div>
      <div id="upload-file-name" style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--color-muted);"></div>
    </div>
    <div class="upload-zone" id="upload-zone">
      <input type="file" id="pcap-input" accept=".pcap">
      <div class="upload-zone-icon">📡</div>
      <div class="upload-zone-text">
        <span>.pcap 파일을 드래그하거나 클릭해서 선택</span><br>
        <span style="color:var(--color-muted);font-size:0.7rem;">업로드 시 기존 세션 테이블이 분석 결과로 교체됩니다</span>
      </div>
    </div>
    <div class="upload-status" id="upload-status"></div>
    <div class="upload-summary" id="upload-summary" style="display:none">
      <div class="upload-stat"><div class="upload-stat-label">Total</div><div class="upload-stat-value us-total" id="us-total">-</div></div>
      <div class="upload-stat"><div class="upload-stat-label">RF Normal</div><div class="upload-stat-value us-normal" id="us-rf-normal">-</div></div>
      <div class="upload-stat"><div class="upload-stat-label">RF Fraud</div><div class="upload-stat-value us-fraud" id="us-rf-fraud">-</div></div>
      <div class="upload-stat"><div class="upload-stat-label">XGB Normal</div><div class="upload-stat-value us-normal" id="us-xgb-normal">-</div></div>
      <div class="upload-stat"><div class="upload-stat-label">XGB Fraud</div><div class="upload-stat-value us-fraud" id="us-xgb-fraud">-</div></div>
      <div class="upload-stat"><div class="upload-stat-label">모델 일치율</div><div class="upload-stat-value us-agree" id="us-agree">-</div></div>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card total">
      <span class="stat-icon">📊</span>
      <div class="stat-label">Total Sessions</div>
      <div class="stat-value" id="stat-total">-</div>
      <div class="stat-sub">분석 완료</div>
    </div>
    <div class="stat-card normal">
      <span class="stat-icon">✅</span>
      <div class="stat-label">Normal</div>
      <div class="stat-value" id="stat-normal">-</div>
      <div class="stat-sub" id="stat-normal-jitter">avg jitter: -</div>
    </div>
    <div class="stat-card fraud">
      <span class="stat-icon">🚨</span>
      <div class="stat-label">Fraud</div>
      <div class="stat-value" id="stat-fraud">-</div>
      <div class="stat-sub" id="stat-fraud-jitter">avg jitter: -</div>
    </div>
    <div class="stat-card ratio">
      <span class="stat-icon">⚠️</span>
      <div class="stat-label">Fraud Ratio</div>
      <div class="stat-value" id="stat-ratio">-</div>
      <div class="stat-sub">전체 대비</div>
    </div>
    <div class="stat-card rf-acc">
      <span class="stat-icon">🌲</span>
      <div class="stat-label">RF Accuracy</div>
      <div class="stat-value" id="stat-rf-acc">-</div>
      <div class="stat-sub">Random Forest</div>
    </div>
    <div class="stat-card xgb-acc">
      <span class="stat-icon">⚡</span>
      <div class="stat-label">XGB Accuracy</div>
      <div class="stat-value" id="stat-xgb-acc">-</div>
      <div class="stat-sub">XGBoost</div>
    </div>
    <div class="stat-card ens-acc">
      <span class="stat-icon">🔗</span>
      <div class="stat-label">Ensemble Acc</div>
      <div class="stat-value" id="stat-ens-acc">-</div>
      <div class="stat-sub" id="stat-agree">RF+XGB Voting</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title"><span>▸</span>Latency Distribution</div>
      <canvas id="latencyChart" height="200"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span>▸</span>Jitter Distribution</div>
      <canvas id="jitterChart" height="200"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span>▸</span>Packet Loss Distribution</div>
      <canvas id="lossChart" height="200"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span>▸</span>Codec Mismatch Distribution</div>
      <canvas id="codecChart" height="200"></canvas>
    </div>
  </div>

  <div class="table-card">
    <div class="table-header">
      <div>
        <div class="chart-title" style="margin-bottom:10px"><span>▸</span>Session Table</div>
        <div class="model-legend">
          <span><span class="legend-dot" style="background:var(--color-rf)"></span>Random Forest</span>
          <span><span class="legend-dot" style="background:var(--color-xgb)"></span>XGBoost</span>
          <span><span class="legend-dot" style="background:var(--color-ens)"></span>Ensemble</span>
          <span style="color:var(--color-muted)">|</span>
          <span id="disagree-count" style="color:#fbbf24">불일치: -</span>
        </div>
      </div>
      <div class="filter-btns">
        <button class="filter-btn active" onclick="filterTable('all',this)">ALL</button>
        <button class="filter-btn" onclick="filterTable('normal',this)">NORMAL</button>
        <button class="filter-btn" onclick="filterTable('fraud',this)">FRAUD</button>
        <button class="filter-btn" onclick="filterTable('disagree',this)">불일치</button>
      </div>
    </div>
    <div style="overflow-x:auto;max-height:420px;overflow-y:auto;">
      <table>
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Label</th>
            <th style="color:var(--color-ens)">Ensemble</th>
            <th style="color:var(--color-ens)">Ens Prob</th>
            <th class="th-rf">RF Prob</th>
            <th class="th-xgb">XGB Prob</th>
            <th>Latency</th>
            <th>Jitter</th>
            <th>Pkt Loss</th>
            <th>Seq Gap</th>
            <th>Codec</th>
          </tr>
        </thead>
        <tbody id="session-table"></tbody>
      </table>
    </div>
  </div>
</main>
<script>
let allSessions=[], jitterChart, latencyChart, lossChart, codecChart, isUploadMode=false;

async function loadStats(){
  const d=await(await fetch('/api/stats')).json();
  document.getElementById('stat-total').textContent=d.total;
  document.getElementById('stat-normal').textContent=d.normal_count;
  document.getElementById('stat-fraud').textContent=d.fraud_count;
  document.getElementById('stat-ratio').textContent=d.total?Math.round(d.fraud_count/d.total*100)+'%':'-';
  document.getElementById('stat-normal-jitter').textContent='avg jitter: '+d.normal_avg_jitter.toFixed(4);
  document.getElementById('stat-fraud-jitter').textContent='avg jitter: '+d.fraud_avg_jitter.toFixed(4);
  document.getElementById('stat-rf-acc').textContent=d.rf_accuracy+'%';
  document.getElementById('stat-xgb-acc').textContent=d.xgb_accuracy+'%';
  document.getElementById('stat-ens-acc').textContent=d.ens_accuracy+'%';
  document.getElementById('stat-agree').textContent='RF+XGB Voting';
  document.getElementById('status-text').textContent='서버 정상 작동 중 (Online)';
}

async function loadSessions(){
  allSessions=await(await fetch('/api/sessions')).json();
  const now=new Date();
  document.getElementById('last-load').textContent=
    '마지막 로드: '+now.toLocaleDateString('ko-KR')+' '+now.toLocaleTimeString('ko-KR');
  const disagreeCount=allSessions.filter(s=>s.rf_label!==s.xgb_label).length;
  document.getElementById('disagree-count').textContent='불일치: '+disagreeCount+'건';
  renderTable(allSessions);
  renderCharts(allSessions);
}

function labelBadge(lbl){
  return lbl===1
    ?'<span class="badge badge-fraud">FRAUD</span>'
    :'<span class="badge badge-normal">NORMAL</span>';
}
function riskBadge(risk){
  if(risk==='HIGH') return '<span class="badge badge-high">HIGH</span>';
  if(risk==='MED')  return '<span class="badge badge-med">MED</span>';
  return '<span class="badge badge-low">LOW</span>';
}
function probBar(prob, cls){
  const pct=Math.round(prob*100);
  return `<div class="prob-bar"><div class="prob-track"><div class="prob-fill ${cls}" style="width:${pct}%"></div></div><span>${prob.toFixed(3)}</span></div>`;
}

function renderTable(sessions){
  document.getElementById('session-table').innerHTML=sessions.slice(0,100).map(s=>{
    const disagree=s.rf_label!==s.xgb_label;
    const rowClass=disagree?'disagree-row':'';
    return `<tr class="${rowClass}">
      <td>${s.session_id.toString().substring(0,24)}${disagree?' <span class="badge badge-disagree">!</span>':''}</td>
      <td>${labelBadge(s.label)}</td>
      <td>${labelBadge(s.ens_label)}</td>
      <td>${probBar(s.ens_prob,'prob-ens')}</td>
      <td>${probBar(s.rf_prob,'prob-rf')}</td>
      <td>${probBar(s.xgb_prob,'prob-xgb')}</td>
      <td>${(s.avg_latency*1000).toFixed(1)}ms</td>
      <td>${(s.avg_jitter*1000).toFixed(1)}ms</td>
      <td>${(s.packet_loss*100).toFixed(1)}%</td>
      <td>${((s.seq_gap_rate||0)*100).toFixed(1)}%</td>
      <td>${(s.codec_mismatch||0).toFixed(3)}</td>
    </tr>`;
  }).join('');
}

function filterTable(type,btn){
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  let filtered=allSessions;
  if(isUploadMode){
    // 업로드 모드: label 없으므로 rf_label 기준으로 필터
    if(type==='normal')   filtered=allSessions.filter(s=>s.rf_label===0);
    else if(type==='fraud')    filtered=allSessions.filter(s=>s.rf_label===1);
    else if(type==='disagree') filtered=allSessions.filter(s=>s.rf_label!==s.xgb_label);
    renderUploadTable(filtered);
  } else {
    if(type==='normal')   filtered=allSessions.filter(s=>s.label===0);
    else if(type==='fraud')    filtered=allSessions.filter(s=>s.label===1);
    else if(type==='disagree') filtered=allSessions.filter(s=>s.rf_label!==s.xgb_label);
    renderTable(filtered);
  }
}

function hist(data, key, min, step, bins=20){
  const c=new Array(bins).fill(0);
  data.forEach(s=>{const i=Math.min(bins-1,Math.floor((s[key]-min)/step));c[i]++;});
  return c;
}

const CHART_OPTS={
  responsive:true,
  plugins:{legend:{labels:{color:'#94a3b8',font:{family:'JetBrains Mono',size:11}}}},
  scales:{
    x:{ticks:{color:'#94a3b8',font:{family:'JetBrains Mono',size:9},maxTicksLimit:6},grid:{color:'#334155'}},
    y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}
  }
};

function renderCharts(sessions){
  const normal=sessions.filter(s=>s.label===0);
  const fraud=sessions.filter(s=>s.label===1);
  const bins=20;

  const allJ=sessions.map(s=>s.avg_jitter);
  const minJ=Math.min(...allJ),maxJ=Math.max(...allJ),stepJ=(maxJ-minJ)/bins;
  const labelsJ=Array.from({length:bins},(_,i)=>(minJ+i*stepJ).toFixed(3));

  if(jitterChart) jitterChart.destroy();
  jitterChart=new Chart(document.getElementById('jitterChart'),{
    type:'bar',
    data:{labels:labelsJ,datasets:[
      {label:'Normal',data:hist(normal,'avg_jitter',minJ,stepJ),backgroundColor:'rgba(34,197,94,0.45)',borderColor:'#22c55e',borderWidth:1},
      {label:'Fraud', data:hist(fraud, 'avg_jitter',minJ,stepJ),backgroundColor:'rgba(239,68,68,0.45)',borderColor:'#ef4444',borderWidth:1}
    ]},
    options:CHART_OPTS
  });

  const allL=sessions.map(s=>s.avg_latency);
  const minL=Math.min(...allL),maxL=Math.max(...allL),stepL=(maxL-minL)/bins;
  const labelsL=Array.from({length:bins},(_,i)=>(minL+i*stepL).toFixed(3));

  if(latencyChart) latencyChart.destroy();
  latencyChart=new Chart(document.getElementById('latencyChart'),{
    type:'bar',
    data:{labels:labelsL,datasets:[
      {label:'Normal',data:hist(normal,'avg_latency',minL,stepL),backgroundColor:'rgba(34,197,94,0.45)',borderColor:'#22c55e',borderWidth:1},
      {label:'Fraud', data:hist(fraud, 'avg_latency',minL,stepL),backgroundColor:'rgba(239,68,68,0.45)',borderColor:'#ef4444',borderWidth:1}
    ]},
    options:CHART_OPTS
  });

  // Packet Loss
  const allPL=sessions.map(s=>s.packet_loss||0);
  const minPL=Math.min(...allPL),maxPL=Math.max(...allPL),stepPL=Math.max((maxPL-minPL)/bins,0.0001);
  const labelsPL=Array.from({length:bins},(_,i)=>(minPL+i*stepPL).toFixed(3));
  if(lossChart) lossChart.destroy();
  lossChart=new Chart(document.getElementById('lossChart'),{
    type:'bar',
    data:{labels:labelsPL,datasets:[
      {label:'Normal',data:hist(normal,'packet_loss',minPL,stepPL),backgroundColor:'rgba(34,197,94,0.45)',borderColor:'#22c55e',borderWidth:1},
      {label:'Fraud', data:hist(fraud, 'packet_loss',minPL,stepPL),backgroundColor:'rgba(239,68,68,0.45)',borderColor:'#ef4444',borderWidth:1}
    ]},
    options:CHART_OPTS
  });

  // Codec Mismatch
  const allCM=sessions.map(s=>s.codec_mismatch||0);
  const minCM=Math.min(...allCM),maxCM=Math.max(...allCM),stepCM=Math.max((maxCM-minCM)/bins,0.0001);
  const labelsCM=Array.from({length:bins},(_,i)=>(minCM+i*stepCM).toFixed(3));
  if(codecChart) codecChart.destroy();
  codecChart=new Chart(document.getElementById('codecChart'),{
    type:'bar',
    data:{labels:labelsCM,datasets:[
      {label:'Normal',data:hist(normal,'codec_mismatch',minCM,stepCM),backgroundColor:'rgba(34,197,94,0.45)',borderColor:'#22c55e',borderWidth:1},
      {label:'Fraud', data:hist(fraud, 'codec_mismatch',minCM,stepCM),backgroundColor:'rgba(239,68,68,0.45)',borderColor:'#ef4444',borderWidth:1}
    ]},
    options:CHART_OPTS
  });
}

loadStats();
loadSessions();

// ── PCAP 업로드 ──────────────────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const pcapInput  = document.getElementById('pcap-input');
const uploadStatus  = document.getElementById('upload-status');
const uploadSummary = document.getElementById('upload-summary');

uploadZone.addEventListener('dragover', e=>{e.preventDefault();uploadZone.classList.add('dragover');});
uploadZone.addEventListener('dragleave', ()=>uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e=>{
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if(file) uploadPcap(file);
});
pcapInput.addEventListener('change', e=>{
  if(e.target.files[0]) uploadPcap(e.target.files[0]);
});

async function uploadPcap(file){
  if(!file.name.toLowerCase().endsWith('.pcap')){
    setStatus('error', '⚠ .pcap 파일만 업로드 가능합니다');
    return;
  }
  document.getElementById('upload-file-name').textContent = file.name;
  setStatus('parsing', '⏳ PCAP 파싱 중... (RTP 세션 추출 + 모델 예측)');
  uploadSummary.style.display='none';

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch('/api/upload', {method:'POST', body:form});
    const data = await res.json();

    if(!res.ok){
      setStatus('error', '❌ ' + (data.error || '오류가 발생했습니다'));
      return;
    }

    const sessions = data.sessions;
    setStatus('done', `✅ ${data.filename} — ${sessions.length}개 세션 분석 완료`);

    // 요약 통계
    const rfNormal  = sessions.filter(s=>s.rf_label===0).length;
    const rfFraud   = sessions.filter(s=>s.rf_label===1).length;
    const xgbNormal = sessions.filter(s=>s.xgb_label===0).length;
    const xgbFraud  = sessions.filter(s=>s.xgb_label===1).length;
    const agree     = sessions.filter(s=>s.rf_label===s.xgb_label).length;

    document.getElementById('us-total').textContent    = sessions.length;
    document.getElementById('us-rf-normal').textContent  = rfNormal;
    document.getElementById('us-rf-fraud').textContent   = rfFraud;
    document.getElementById('us-xgb-normal').textContent = xgbNormal;
    document.getElementById('us-xgb-fraud').textContent  = xgbFraud;
    document.getElementById('us-agree').textContent = Math.round(agree/sessions.length*100)+'%';
    uploadSummary.style.display='flex';

    // 테이블 교체 (업로드 결과로 대체)
    allSessions = sessions;
    isUploadMode = true;
    const disagreeCount = sessions.filter(s=>s.rf_label!==s.xgb_label).length;
    document.getElementById('disagree-count').textContent='불일치: '+disagreeCount+'건';
    renderUploadTable(sessions);
    renderUploadCharts(sessions);

  } catch(err){
    setStatus('error', '❌ 네트워크 오류: ' + err.message);
  }
}

function setStatus(type, msg){
  uploadStatus.className = 'upload-status ' + type;
  uploadStatus.textContent = msg;
}

function renderUploadTable(sessions){
  // 업로드 모드: label 컬럼 없이 앙상블+RF/XGB 예측 표시
  document.getElementById('session-table').innerHTML = sessions.slice(0,100).map(s=>{
    const disagree = s.rf_label !== s.xgb_label;
    const rowClass = disagree ? 'disagree-row' : '';
    return `<tr class="${rowClass}">
      <td>${s.session_id.toString().substring(0,24)}${disagree?' <span class="badge badge-disagree">!</span>':''}</td>
      <td><span style="color:var(--color-muted);font-size:0.68rem;font-family:'JetBrains Mono',monospace;">UPLOAD</span></td>
      <td>${labelBadge(s.ens_label)}</td>
      <td>${probBar(s.ens_prob,'prob-ens')}</td>
      <td>${probBar(s.rf_prob,'prob-rf')}</td>
      <td>${probBar(s.xgb_prob,'prob-xgb')}</td>
      <td>${(s.avg_latency*1000).toFixed(1)}ms</td>
      <td>${(s.avg_jitter*1000).toFixed(1)}ms</td>
      <td>${(s.packet_loss*100).toFixed(1)}%</td>
      <td>${((s.seq_gap_rate||0)*100).toFixed(1)}%</td>
      <td>${(s.codec_mismatch||0).toFixed(3)}</td>
    </tr>`;
  }).join('');
}

function renderUploadCharts(sessions){
  const fraud  = sessions.filter(s=>s.rf_label===1 || s.xgb_label===1);
  const normal = sessions.filter(s=>s.rf_label===0 && s.xgb_label===0);
  renderCharts([...normal.map(s=>({...s,label:0})), ...fraud.map(s=>({...s,label:1}))]);
}
</script>
</body>
</html>"""

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
