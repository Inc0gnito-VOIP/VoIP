# Inc0gnito — 차세대 SIM Box 탐지 시스템

> **물리적 계층 아티팩트 분석 및 하이브리드 머신러닝 기반 VoIP 우회 사기 탐지**

## 프로젝트 개요

SIM Box는 국제전화를 VoIP로 수신한 뒤 현지 SIM 카드를 통해 국내전화로 위장 발신하는 장비입니다. 보이스피싱의 핵심 인프라로 사용되며, 통신사의 국제 interconnect 수익을 우회합니다.

기존 탐지 방법(CDR 분석, SIM 번호 패턴)은 SIM 카드 교체나 IMEI 조작 등 소프트웨어적 위장에 취약합니다. 본 프로젝트는 **소프트웨어로 은폐할 수 없는 물리적 계층 아티팩트**(RTP 패킷의 지연, 지터, 손실률)를 분석하여 SIM Box를 탐지하는 새로운 접근법을 제안합니다.

### 핵심 가설

> "SIM Box를 경유하는 VoIP 통화는 저품질 하드웨어와 불안정한 인터넷망 특성상 RTP 패킷의 지연(Latency)과 지터(Jitter)가 높아지며, 이는 소프트웨어적으로 은폐하기 어렵다."

## 시스템 구성

| 모듈 | 설명 |
|------|------|
| **The Generator** | Python Scapy로 정상/SIM Box RTP 패킷을 직접 생성. Linux Network Namespaces + veth pair로 격리된 네트워크 환경 구현 |
| **The Sniffer** | `rtp_parser.py` — PCAP에서 세션별 6개 피처 추출 (RFC 3550 지터, 시퀀스 갭, 코덱 변경 등) |
| **The Brain** | Random Forest + XGBoost + VotingClassifier(Soft Voting) 앙상블. Hard Examples 테스트로 현실적 평가 |
| **The Dashboard** | Flask 기반 실시간 관제 대시보드. PCAP 업로드 → 자동 분석 → RF/XGB/Ensemble 3모델 비교 |

## 탐지 피처 (6개)

| 피처 | 분류 | 설명 |
|------|------|------|
| `avg_latency` | 물리 계층 | 패킷 평균 도착 간격 (초) |
| `avg_jitter` | 물리 계층 | RFC 3550 기준 평균 지터 (초) |
| `iat_variance` | 물리 계층 | 패킷 도착 간격 분산 |
| `packet_loss` | 행동 피처 | 시퀀스 번호 갭 기반 손실률 |
| `seq_gap_rate` | 행동 피처 | 비정상 시퀀스 번호 비율 |
| `codec_mismatch` | 행동 피처 | 세션 중 코덱 변경 비율 |

## SIM Box 시나리오 (4타입)

현실의 SIM Box 운영 환경을 반영하여 4가지 난이도별 시나리오를 설계했습니다.

| 타입 | Latency | 특성 | 탐지 난이도 |
|------|---------|------|-------------|
| **Obvious** | 35ms | 고지터 + 패킷손실 + 코덱전환 | 쉬움 |
| **Stealth** | 15~25ms | 행동 피처 정상 위장, 물리 피처로만 탐지 | 중간 |
| **Mixed** | 20~35ms | 일부 피처만 위장 | 중간 |
| **Premium** | 10~20ms | 고급 장비 + 저지연 경로, 정상과 겹침 | 어려움 |

## 실험 결과

### 학습 데이터
- 6,739세션 (정상 1,739 + SIM Box 5,000)
- SIM Box 4타입 각 25% 균등

### 5-Fold Cross Validation

| 모델 | F1 (mean ± std) |
|------|-----------------|
| Random Forest | 0.9961 ± 0.0012 |
| XGBoost | 0.9955 ± 0.0013 |
| Ensemble (Voting) | 0.9960 ± 0.0011 |

### Hard Examples 테스트 (1,000세션)

테스트 데이터 구성: 정상 700 + SIM Box 300 (Obvious 10% / Stealth 30% / Mixed 30% / Premium 30%)

| 모델 | F1 | Precision | Recall | FPR |
|------|-----|-----------|--------|------|
| RF | 0.8258 | 1.00 | 0.70 | 0.0000 |
| XGB | 0.8743 | 0.99 | 0.78 | 0.0014 |
| Ensemble | 0.8701 | 0.99 | 0.77 | 0.0014 |

### 타입별 탐지율 (Ensemble)

| 타입 | 탐지율 |
|------|--------|
| Obvious | 100% |
| Stealth | 63.1% |
| Mixed | ~88% |
| Premium | 43.4% |

**핵심 성과:** FPR ≈ 0 (정상 통화 오탐 거의 없음). Feature Importance에서 물리 계층 피처(avg_latency, iat_variance)가 1~2위를 차지하여 핵심 가설을 실험으로 검증.

## 기술 스택

| 분야 | 기술 |
|------|------|
| 언어 | Python 3.8+ |
| 패킷 생성/파싱 | Scapy |
| 데이터 처리 | Pandas, NumPy |
| ML 모델 | scikit-learn (Random Forest), XGBoost, VotingClassifier |
| 웹 서버 | Flask + REST API |
| DB | SQLite |
| 시뮬레이션 | Linux Network Namespaces + veth pair (VirtualBox VM) |
| 예산 | 0원 (CPU 기반, GPU 불필요) |

## 프로젝트 구조

```
VoIP/
├── src/
│   ├── app/app.py          # Flask 서버 + 대시보드 (HTML 내장)
│   ├── parser/rtp_parser.py # PCAP → 6피처 CSV 추출
│   └── model/model.pkl      # 학습된 Ensemble 모델
├── data/
│   ├── raw/                 # PCAP 파일
│   └── processed/           # sessions.csv, test_sessions.csv
├── sender.py                # RTP 패킷 생성기 (정상 + SIM Box 4타입)
├── test_sender.py           # 테스트 데이터 생성기 (Hard Examples)
├── train.py                 # RF + XGB + Voting 앙상블 학습
├── eval_test.py             # 테스트 데이터 평가
├── docs/index.html          # GitHub Pages 현황판
└── requirements.txt
```

## 실행 방법

```bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. 서버 실행
python src/app/app.py
# → http://127.0.0.1:5000

# 3. 브라우저에서 대시보드 접속 → PCAP 파일 업로드 → 자동 분석
```

## 팀원 및 역할

| 팀원 | 역할 | 주요 담당 |
|------|------|-----------|
| 현서 | SW/웹 개발 | Flask 대시보드, 파서, GitHub 관리, 문서 |
| 광준 | AI 모델러 | ML 모델 학습, Feature Engineering, 성능 평가 |
| 이안 | 네트워크/데이터 | 데이터 생성, 인프라, 네트워크 시뮬레이션 |

## 참고 자료

- GSMA × Virginia Tech BRICCS, "SIM Farm Detection," MWC Barcelona 2026
- SigN: SIMBox Activity Detection Through Latency Anomalies at the Cellular Edge (ACM ASIA CCS 2025)
- Subex, "SIMBox Fraud: Challenges and AI-Powered Solutions for Telecom Operators"
- Boxed Out: Blocking Cellular Interconnect Bypass Fraud (USENIX Security 2015)

## 링크

- **GitHub Pages 현황판:** https://grute02.github.io/VoIP/
- **GitHub 저장소:** https://github.com/grute02/VoIP
