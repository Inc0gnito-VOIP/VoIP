# CLAUDE.md — Inc0gnito 프로젝트 컨텍스트

## 프로젝트 개요

**프로젝트명:** 물리적 계층 아티팩트 분석 및 하이브리드 머신러닝 기반 차세대 SIM Box 탐지 시스템  
**팀명:** Inc0gnito  
**GitHub:** https://github.com/grute02/VoIP  
**현황판:** https://grute02.github.io/VoIP/  
**최종 마감:** 2026-04-26 | **컨퍼런스:** 2026-05-09

---

## 팀원 및 역할

| 팀원 | 역할 | 주요 담당 |
|------|------|-----------|
| 현서 (isoft) | 소프트웨어/웹 개발 | 파서, Flask 대시보드, GitHub 관리, 문서 |
| 광준 | AI 모델러 | ML 모델 학습, Feature Engineering, 성능 평가 |
| 이안 | 네트워크/데이터 | 데이터 생성, 인프라, tc-netem 시나리오 |

---

## 핵심 가설

> "SIM Box를 경유하는 VoIP 통화는 저품질 하드웨어와 불안정한 인터넷망 특성상 RTP 패킷의 지연(Latency)과 지터(Jitter)가 높아지며, 이는 소프트웨어적으로 은폐하기 어렵다."

---

## 기술 스택

- **언어:** Python 3.8+
- **패킷 생성/파싱:** Scapy
- **데이터 처리:** Pandas
- **ML:** scikit-learn (Random Forest), XGBoost
- **웹 서버:** Flask
- **DB:** SQLite
- **시뮬레이션:** Linux tc-netem
- **인프라:** Ubuntu 22.04 (WSL2), Wireshark

---

## 데이터 파이프라인

```
Python Scapy로 RTP 패킷 직접 생성
    ↓
PCAP 파일 저장 (data/raw/)
    ↓
rtp_parser.py로 세션별 피처 추출
    ↓
sessions.csv (data/processed/)
    ↓
Random Forest + XGBoost 학습
    ↓
model.pkl → Flask /api/analyze 연동
    ↓
대시보드 시각화
```

---

## 피처 정의 (sessions.csv 컬럼)

| 컬럼 | 설명 |
|------|------|
| session_id | 세션 식별자 |
| avg_latency | 패킷 평균 도착 간격 (초) |
| avg_jitter | RFC 3550 기준 평균 지터 (초) |
| iat_variance | 패킷 도착 간격 분산 |
| packet_loss | 시퀀스 번호 갭 기반 손실률 |
| seq_gap_rate | 비정상 시퀀스 번호 비율 |
| packet_count | 세션당 패킷 수 |
| label | 0=정상, 1=사기 |

**현재 데이터셋:** 8,000세션 (정상 4,000 / 사기 4,000)  
**생성 방식:** 정상/사기 모두 Python Scapy로 동일 메커니즘 생성 (데이터 누수 없음)

```
정상: gauss(0.004, 0.002)  → avg_latency ≈ 4ms
사기: gauss(0.15,  0.05)   → avg_latency ≈ 150ms
```

---

## 프로젝트 구조

```
VoIP/
├── data/
│   ├── raw/          # PCAP 파일 (정상/사기)
│   └── processed/    # sessions.csv (.gitignore 제외)
├── src/
│   ├── parser/       # rtp_parser.py
│   ├── model/        # model.pkl (학습된 모델)
│   └── app/          # app.py (Flask 서버)
├── docs/             # GitHub Pages 현황판 + 참조 문서
│   ├── index.html    # 현황판 (GitHub Pages 배포)
│   ├── CLAUDE.md     # 이 파일
│   └── ...           # 계획서, 리비전 관련 문서들
├── requirements.txt
└── README.md
```

---

## Flask API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | /api/sessions | 전체 세션 목록 |
| GET | /api/stats | 통계 요약 |
| POST | /api/analyze | 단일 세션 실시간 탐지 |
| POST | /api/analyze/batch | 전체 세션 일괄 탐지 |

### /api/analyze 요청 예시
```json
{
  "avg_latency": 0.004,
  "avg_jitter": 0.005,
  "iat_variance": 0.000004,
  "packet_loss": 0.0,
  "seq_gap_rate": 0.046
}
```

---

## 서버 실행 방법

```bash
cd ~/Documents/Q.E.D/프젝/Incognito_Project/VoIP
pip install flask numpy scikit-learn
python src/app/app.py
# → http://127.0.0.1:5000
```

---

## 주요 결정사항 (변경 이력)

| 항목 | 원래 계획 | 실제 구현 | 변경 이유 |
|------|-----------|-----------|-----------|
| 데이터 생성 | SIPp + Asterisk + tcpdump | Python Scapy 직접 생성 | SIPp RTP 미포함 문제 |
| 대시보드 | Streamlit | Flask | REST API 설계 및 팀 분업 필요 |
| 데이터 생성 도구 | tc-netem | Python gauss() | 동일 메커니즘으로 누수 방지 |
| SMOTE | 적용 예정 | 불필요 | 데이터가 이미 4:4 균형 |

---

## 현재 진행 상황 (2026-03-19 기준, 6주차)

- [x] Flask 서버 + SQLite DB 구축
- [x] rtp_parser.py 완성 (RFC 3550 지터)
- [x] sessions.csv 8,000세션 생성
- [x] model.pkl 연동 + /api/analyze 완성
- [x] 대시보드 MVP (Chart.js + Risk Score)
- [x] 2차 리비전 완료
- [ ] 광준: RF 베이스라인 학습 (이안 PCAP + 현서 CSV 병행)
- [ ] 현서: 2차 리비전 답변 작성
- [ ] 이안/현서: 혼잡도 변수 유효성 조사
- [ ] 광준: model.pkl 완성 후 현서에게 전달

---

## 2차 리비전 피드백 요약

**리뷰어 B:** PoC로 빠르게 확인 → 현재 방향과 일치, 그대로 진행  
**리뷰어 C-1:** 혼잡도 변수 추가 권고 → 이안/현서 유효성 조사 중  
**리뷰어 C-2:** Flask+SQLite 스케일 한계 (선택사항) → 현재 스킵, 발표에서 향후 과제로 언급

---

## 관련 외부 연구

**Virginia Tech BRICCS × GSMA (MWC 2026)**  
- 접근법: 시그널링/RAN 계층 분석 (기지국 KPI, SIM 간 상관관계)  
- 우리와 차이: 우리는 RTP 물리 계층 분석 → 통신사 내부 데이터 불필요, 조작 불가  
- 의의: 동일한 "시뮬레이션 + ML" 방법론을 글로벌 연구팀도 사용 중

---

## GPU 관련

Random Forest + XGBoost는 CPU 기반 알고리즘. GPU 불필요.  
8,000세션 학습 시간: 약 0.26초 (로컬 CPU 기준)  
예측(추론)도 즉각적. 0원 예산 계획 유지.

---

## Claude Code 작업 가이드

### 파일 위치 요약

| 파일 | 경로 | 설명 |
|------|------|------|
| Flask 서버 | `src/app/app.py` | API + 대시보드 HTML 포함 |
| 현황판 | `docs/index.html` | GitHub Pages 배포본 |
| 계획서 | `docs/` 폴더 참조 | 프로젝트 계획서, 리비전 관련 문서 |
| 파서 | `src/parser/rtp_parser.py` | PCAP → CSV 변환 |
| 모델 | `src/model/model.pkl` | 학습된 RF 모델 |
| 데이터 | `data/processed/sessions.csv` | .gitignore 제외 |

### 대시보드 (src/app/app.py) 수정 시 주의사항

- HTML이 `DASHBOARD_HTML` 문자열 변수 안에 전부 포함되어 있음
- Chart.js 버전: 4.4.0 (cdnjs)
- 폰트: JetBrains Mono (코드), Syne (제목)
- CSS 변수 (색상):
  ```
  --bg: #0a0e1a       배경
  --surface: #111827  카드 배경
  --border: #1e2d40   테두리
  --accent: #00d4ff   강조 (파란색)
  --danger: #ff4d6d   위험 (빨간색)
  --success: #00f5a0  정상 (초록색)
  --text: #e2e8f0     본문
  --muted: #64748b    보조 텍스트
  ```
- API 엔드포인트: `/api/sessions`, `/api/stats`, `/api/analyze`, `/api/analyze/batch`
- model.pkl 경로: `BASE_DIR / 'src' / 'model' / 'model.pkl'`

### 현황판 (docs/index.html) 수정 시 주의사항

- 상태값은 localStorage에 저장됨 (브라우저별 독립적, GitHub에 저장 안 됨)
- WBS 상태 클래스: `rs-done`, `rs-cur`, `rs-pend`
- 태스크 상태 클래스: `s-done`, `s-wip`, `s-pend`, `s-block`
- 멤버 배지 클래스: `msb-done`, `msb-wip`, `msb-block`
- 현재 주차 표시: `wcur` 클래스 + `wn`에 `▶` 추가
- 블로커 감지: `data-blocker` 속성이 있는 `.ti.s-block` 요소 자동 집계
- CSS 변수:
  ```
  --done: #34d399     완료 (초록)
  --wip:  #fbbf24     진행중 (노란)
  --block: #f87171    블로커 (빨간)
  --pend: #475569     예정 (회색)
  --a: #22d3ee        이안 색상
  --b: #a78bfa        광준 색상
  --c: #34d399        현서 색상
  ```

### 서버 실행

```bash
cd ~/Documents/Q.E.D/프젝/Incognito_Project/VoIP
python src/app/app.py
# → http://127.0.0.1:5000
```

### GitHub 배포 (현황판)

```bash
# docs/index.html 수정 후
git add docs/index.html
git commit -m "docs: 현황판 업데이트"
git push
# → https://grute02.github.io/VoIP/ 에 반영
```
