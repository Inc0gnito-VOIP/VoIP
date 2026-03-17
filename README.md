# 🛡️ 차세대 SIM Box 탐지 시스템 구축 프로젝트

> **부제:** RTP 패킷의 시간적 특성 분석을 통한 VoIP 우회 사기 탐지 프로토타입 개발

## 1. 프로젝트 개요

* **배경:** 고도화되는 SIM Box 사기로 인한 통신사 수익 악화 및 국가적 세수 손실 대응.
* **핵심 가설:** "물리적 아티팩트(지연, 지터 등)는 소프트웨어적으로 완벽히 은폐하기 어렵다".
* **최종 목표:** 90% 이상의 탐지 정확도를 갖는 경량화된 보안 시스템 프로토타입 구축.

## 2. 주요 기능

* **The Generator:** Python Scapy 기반 가상 VoIP 트래픽 직접 생성 (정상/사기 시나리오).
* **The Sniffer:** Python Scapy 기반의 RTP 헤더 파싱 및 지터(Jitter) 계산.
* **The Brain:** Random Forest 및 XGBoost 기반의 하이브리드 탐지 엔진.
* **The Dashboard:** Flask 기반의 실시간 위험도 시각화 관제 대시보드.

## 3. 기술 스택

* **Language:** Python 3.8+
* **Libraries:** Scapy, Pandas, Scikit-learn, XGBoost, Flask
* **Tools:** Ubuntu 22.04 LTS, tc-netem, Wireshark

## 4. 팀원 및 역할 (R&R)

* **이안 (팀원 A):** 네트워크 및 데이터 엔지니어 (데이터 생성 및 환경 관리).
* **광준 (팀원 B):** AI 모델러 및 분석가 (탐지 알고리즘 및 머신러닝 설계).
* **현서 (팀원 C):** 소프트웨어 및 웹 개발자 (시스템 통합 및 시각화 구현).
