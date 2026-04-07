from flask import Flask, jsonify, render_template_string
import csv
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
CSV_PATH = BASE_DIR / 'data' / 'processed' / 'sessions.csv'


def load_sessions():
    sessions = []
    if not CSV_PATH.exists():
        return sessions
    with open(CSV_PATH, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sessions.append({
                'session_id': row['session_id'],
                'avg_latency': float(row['avg_latency']),
                'avg_jitter': float(row['avg_jitter']),
                'iat_variance': float(row['iat_variance']),
                'packet_loss': float(row['packet_loss']),
                'seq_gap_rate': float(row['seq_gap_rate']),
                'packet_count': int(row['packet_count']),
                'label': int(row['label']),
            })
    return sessions


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
        return jsonify({'total': 0})
    normal = [s for s in sessions if s['label'] == 0]
    fraud  = [s for s in sessions if s['label'] == 1]

    def avg(lst, key):
        return round(sum(s[key] for s in lst) / len(lst), 6) if lst else 0

    return jsonify({
        'total': len(sessions),
        'normal_count': len(normal),
        'fraud_count': len(fraud),
        'normal_avg_jitter': avg(normal, 'avg_jitter'),
        'fraud_avg_jitter': avg(fraud, 'avg_jitter'),
        'normal_avg_latency': avg(normal, 'avg_latency'),
        'fraud_avg_latency': avg(fraud, 'avg_latency'),
    })


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
    --bg:#0a0e1a; --surface:#111827; --border:#1e2d40;
    --accent:#00d4ff; --danger:#ff4d6d; --success:#00f5a0;
    --text:#e2e8f0; --muted:#64748b;
  }
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Syne',sans-serif;min-height:100vh;}
  header{padding:24px 32px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;}
  header h1{font-size:1.4rem;font-weight:800;letter-spacing:-0.02em;}
  header h1 span{color:var(--accent);}
  .status-dot{width:8px;height:8px;background:var(--success);border-radius:50%;display:inline-block;margin-right:8px;box-shadow:0 0 8px var(--success);animation:pulse 2s infinite;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
  .status-text{font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--success);}
  main{padding:32px;max-width:1400px;margin:0 auto;}
  .stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px;}
  .stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;position:relative;overflow:hidden;}
  .stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
  .stat-card.total::before{background:var(--accent);}
  .stat-card.normal::before{background:var(--success);}
  .stat-card.fraud::before{background:var(--danger);}
  .stat-card.ratio::before{background:#a78bfa;}
  .stat-label{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;font-family:'JetBrains Mono',monospace;}
  .stat-value{font-size:2rem;font-weight:800;line-height:1;}
  .stat-card.total .stat-value{color:var(--accent);}
  .stat-card.normal .stat-value{color:var(--success);}
  .stat-card.fraud .stat-value{color:var(--danger);}
  .stat-card.ratio .stat-value{color:#a78bfa;}
  .stat-sub{font-size:0.75rem;color:var(--muted);margin-top:6px;font-family:'JetBrains Mono',monospace;}
  .charts-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:32px;}
  .chart-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;}
  .chart-title{font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:20px;font-family:'JetBrains Mono',monospace;}
  .chart-title span{color:var(--accent);margin-right:8px;}
  .table-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;}
  .table-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;}
  .filter-btns{display:flex;gap:8px;}
  .filter-btn{padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;font-family:'JetBrains Mono',monospace;font-size:0.72rem;transition:all 0.2s;}
  .filter-btn.active,.filter-btn:hover{border-color:var(--accent);color:var(--accent);}
  table{width:100%;border-collapse:collapse;}
  th{text-align:left;padding:10px 14px;font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;}
  td{padding:12px 14px;font-size:0.82rem;font-family:'JetBrains Mono',monospace;border-bottom:1px solid rgba(30,45,64,0.5);}
  tr:last-child td{border-bottom:none;}
  tr:hover td{background:rgba(255,255,255,0.02);}
  .badge{display:inline-block;padding:3px 10px;border-radius:4px;font-size:0.68rem;font-weight:700;letter-spacing:0.05em;}
  .badge-normal{background:rgba(0,245,160,0.1);color:var(--success);border:1px solid rgba(0,245,160,0.2);}
  .badge-fraud{background:rgba(255,77,109,0.1);color:var(--danger);border:1px solid rgba(255,77,109,0.2);}
  .jitter-bar{display:flex;align-items:center;gap:8px;}
  .jitter-track{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden;}
  .jitter-fill{height:100%;border-radius:2px;}
</style>
</head>
<body>
<header>
  <h1>INC0GNITO <span>//</span> SIM Box Detection</h1>
  <div><span class="status-dot"></span><span class="status-text" id="status-text">서버 연결 중...</span></div>
</header>
<main>
  <div class="stats-grid">
    <div class="stat-card total">
      <div class="stat-label">Total Sessions</div>
      <div class="stat-value" id="stat-total">-</div>
      <div class="stat-sub">분석 완료</div>
    </div>
    <div class="stat-card normal">
      <div class="stat-label">Normal</div>
      <div class="stat-value" id="stat-normal">-</div>
      <div class="stat-sub" id="stat-normal-jitter">avg jitter: -</div>
    </div>
    <div class="stat-card fraud">
      <div class="stat-label">Fraud</div>
      <div class="stat-value" id="stat-fraud">-</div>
      <div class="stat-sub" id="stat-fraud-jitter">avg jitter: -</div>
    </div>
    <div class="stat-card ratio">
      <div class="stat-label">Fraud Ratio</div>
      <div class="stat-value" id="stat-ratio">-</div>
      <div class="stat-sub">전체 대비</div>
    </div>
  </div>
  <div class="charts-grid">
    <div class="chart-card">
      <div class="chart-title"><span>▸</span>Jitter Distribution</div>
      <canvas id="jitterChart" height="200"></canvas>
    </div>
    <div class="chart-card">
      <div class="chart-title"><span>▸</span>Latency Distribution</div>
      <canvas id="latencyChart" height="200"></canvas>
    </div>
  </div>
  <div class="table-card">
    <div class="table-header">
      <div class="chart-title" style="margin:0"><span>▸</span>Session Table</div>
      <div class="filter-btns">
        <button class="filter-btn active" onclick="filterTable('all',this)">ALL</button>
        <button class="filter-btn" onclick="filterTable('normal',this)">NORMAL</button>
        <button class="filter-btn" onclick="filterTable('fraud',this)">FRAUD</button>
      </div>
    </div>
    <div style="overflow-x:auto;max-height:400px;overflow-y:auto;">
      <table>
        <thead><tr><th>Session ID</th><th>Label</th><th>Avg Jitter</th><th>Avg Latency</th><th>Packet Loss</th><th>Seq Gap Rate</th><th>Packets</th></tr></thead>
        <tbody id="session-table"></tbody>
      </table>
    </div>
  </div>
</main>
<script>
let allSessions=[], jitterChart, latencyChart;

async function loadStats(){
  const d=await(await fetch('/api/stats')).json();
  document.getElementById('stat-total').textContent=d.total;
  document.getElementById('stat-normal').textContent=d.normal_count;
  document.getElementById('stat-fraud').textContent=d.fraud_count;
  document.getElementById('stat-ratio').textContent=d.total?Math.round(d.fraud_count/d.total*100)+'%':'-';
  document.getElementById('stat-normal-jitter').textContent='avg jitter: '+d.normal_avg_jitter.toFixed(4);
  document.getElementById('stat-fraud-jitter').textContent='avg jitter: '+d.fraud_avg_jitter.toFixed(4);
  document.getElementById('status-text').textContent='서버 정상 작동 중 (Online)';
}

async function loadSessions(){
  allSessions=await(await fetch('/api/sessions')).json();
  renderTable(allSessions);
  renderCharts(allSessions);
}

function renderTable(sessions){
  const maxJ=Math.max(...sessions.map(s=>s.avg_jitter));
  document.getElementById('session-table').innerHTML=sessions.slice(0,50).map(s=>{
    const pct=Math.min(100,(s.avg_jitter/maxJ*100)).toFixed(1);
    const color=s.label===1?'#ff4d6d':'#00f5a0';
    const badge=s.label===1
      ?'<span class="badge badge-fraud">FRAUD</span>'
      :'<span class="badge badge-normal">NORMAL</span>';
    return `<tr>
      <td>${s.session_id.substring(0,28)}</td>
      <td>${badge}</td>
      <td><div class="jitter-bar"><div class="jitter-track"><div class="jitter-fill" style="width:${pct}%;background:${color}"></div></div><span>${s.avg_jitter.toFixed(4)}</span></div></td>
      <td>${s.avg_latency.toFixed(4)}s</td>
      <td>${(s.packet_loss*100).toFixed(1)}%</td>
      <td>${(s.seq_gap_rate*100).toFixed(1)}%</td>
      <td>${s.packet_count}</td>
    </tr>`;
  }).join('');
}

function filterTable(type,btn){
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  renderTable(type==='all'?allSessions:allSessions.filter(s=>s.label===(type==='fraud'?1:0)));
}

function hist(data, key, min, step, bins=20){
  const c=new Array(bins).fill(0);
  data.forEach(s=>{const i=Math.min(bins-1,Math.floor((s[key]-min)/step));c[i]++;});
  return c;
}

function renderCharts(sessions){
  const normal=sessions.filter(s=>s.label===0);
  const fraud=sessions.filter(s=>s.label===1);
  const bins=20;

  const allJ=sessions.map(s=>s.avg_jitter);
  const minJ=Math.min(...allJ), maxJ=Math.max(...allJ), stepJ=(maxJ-minJ)/bins;
  const labelsJ=Array.from({length:bins},(_,i)=>(minJ+i*stepJ).toFixed(3));

  if(jitterChart) jitterChart.destroy();
  jitterChart=new Chart(document.getElementById('jitterChart'),{
    type:'bar',
    data:{labels:labelsJ,datasets:[
      {label:'Normal',data:hist(normal,'avg_jitter',minJ,stepJ),backgroundColor:'rgba(0,245,160,0.5)',borderColor:'#00f5a0',borderWidth:1},
      {label:'Fraud', data:hist(fraud, 'avg_jitter',minJ,stepJ),backgroundColor:'rgba(255,77,109,0.5)',borderColor:'#ff4d6d',borderWidth:1}
    ]},
    options:{responsive:true,
      plugins:{legend:{labels:{color:'#64748b',font:{family:'JetBrains Mono',size:11}}}},
      scales:{
        x:{ticks:{color:'#64748b',font:{family:'JetBrains Mono',size:9},maxTicksLimit:6},grid:{color:'#1e2d40'}},
        y:{ticks:{color:'#64748b'},grid:{color:'#1e2d40'}}
      }
    }
  });

  const allL=sessions.map(s=>s.avg_latency);
  const minL=Math.min(...allL), maxL=Math.max(...allL), stepL=(maxL-minL)/bins;
  const labelsL=Array.from({length:bins},(_,i)=>(minL+i*stepL).toFixed(3));

  if(latencyChart) latencyChart.destroy();
  latencyChart=new Chart(document.getElementById('latencyChart'),{
    type:'bar',
    data:{labels:labelsL,datasets:[
      {label:'Normal',data:hist(normal,'avg_latency',minL,stepL),backgroundColor:'rgba(0,212,255,0.5)',borderColor:'#00d4ff',borderWidth:1},
      {label:'Fraud', data:hist(fraud, 'avg_latency',minL,stepL),backgroundColor:'rgba(167,139,250,0.5)',borderColor:'#a78bfa',borderWidth:1}
    ]},
    options:{responsive:true,
      plugins:{legend:{labels:{color:'#64748b',font:{family:'JetBrains Mono',size:11}}}},
      scales:{
        x:{ticks:{color:'#64748b',font:{family:'JetBrains Mono',size:9},maxTicksLimit:6},grid:{color:'#1e2d40'}},
        y:{ticks:{color:'#64748b'},grid:{color:'#1e2d40'}}
      }
    }
  });
}

loadStats();
loadSessions();
</script>
</body>
</html>"""

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
