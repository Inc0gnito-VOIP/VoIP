# 🛡️ 차세대 SIM Box 탐지 시스템 구축 프로젝트
> **부제:** RTP 패킷의 시간적 특성 분석을 통한 VoIP 우회 사기 탐지 프로토타입 개발

## 1. 프로젝트 개요
- **배경:** 고도화되는 SIM Box 사기로 인한 통신사 수익 악화 및 국가적 세수 손실 대응.
- **핵심 가설:** "물리적 아티팩트(지연, 지터 등)는 소프트웨어적으로 완벽히 은폐하기 어렵다".
- **최종 목표:** 90% 이상의 탐지 정확도를 갖는 경량화된 보안 시스템 프로토타입 구축.

## 2. 주요 기능
- **The Generator:** Linux tc-netem을 활용한 가상 VoIP 트래픽 생성 및 환경 모사.
- **The Sniffer:** Python Scapy 기반의 RTP 헤더 파싱 및 지터(Jitter) 계산.
- **The Brain:** Random Forest 및 XGBoost 기반의 하이브리드 탐지 엔진.
- **The Dashboard:** Streamlit 기반의 실시간 위험도 시각화 관제 대시보드.

## 3. 기술 스택
- **Language:** Python 3.8+
- **Libraries:** Scapy, Pandas, Scikit-learn, XGBoost, Streamlit
- **Tools:** Ubuntu 22.04 LTS, tc-netem, SIPp, Wireshark

## 4. 팀원 및 역할 (R&R)
- **이안 (팀원 A):** 네트워크 및 데이터 엔지니어 (데이터 생성 및 환경 관리).
- **광준 (팀원 B):** AI 모델러 및 분석가 (탐지 알고리즘 및 머신러닝 설계).
- **현서 (팀원 C):** 소프트웨어 및 웹 개발자 (시스템 통합 및 시각화 구현).

## 🛠️ 프로젝트 환경 설정 가이드 (Team Only)

모든 팀원은 원활한 협업을 위해 아래 단계를 순서대로 따라 환경 설정을 완료해 주세요.

### 1. 필수 시스템 패키지 설치
터미널을 열고 파이썬 가상 환경 구축과 패킷 분석에 필요한 도구들을 설치합니다.

```bash
sudo apt update
sudo apt install python3-pip python3-venv tshark -y
```

### 2. 가상 환경(venv) 생성 및 활성화
프로젝트 폴더 내에서 독립된 개발 환경을 구축합니다. (가상 환경 폴더는 .gitignore에 의해 관리 대상에서 제외됩니다.)

```bash
# 가상 환경 생성 (최초 1회 실행)
python3 -m venv venv

# 가상 환경 활성화 (작업 시작 시 매번 실행)
source venv/bin/activate
```
활성화 성공 시 터미널 프롬프트 앞에 (venv) 표시가 나타납니다.

### 3. 라이브러리 설치
프로젝트에 필요한 핵심 라이브러리들을 한 번에 설치합니다. 

```bash
# pip 최신 버전 업데이트
pip install --upgrade pip

# requirements.txt에 명시된 라이브러리 설치
pip install -r requirements.txt
```
