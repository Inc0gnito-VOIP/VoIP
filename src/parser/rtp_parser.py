"""
rtp_parser.py
PCAP 파일에서 RTP 패킷을 파싱하여 세션별 특징(feature)을 추출하고 CSV로 저장
"""

import sys
import struct
import csv
from pathlib import Path

try:
    from scapy.all import rdpcap
    from scapy.layers.inet import UDP
except ImportError:
    print("scapy가 설치되어 있지 않습니다. pip install scapy 를 실행하세요.")
    sys.exit(1)


def parse_rtp_header(payload: bytes):
    if len(payload) < 12:
        return None
    try:
        byte0, byte1 = payload[0], payload[1]
        version = (byte0 >> 6) & 0x03
        if version != 2:
            return None
        seq = struct.unpack('!H', payload[2:4])[0]
        timestamp = struct.unpack('!I', payload[4:8])[0]
        ssrc = struct.unpack('!I', payload[8:12])[0]
        return {'seq': seq, 'timestamp': timestamp, 'ssrc': ssrc}
    except Exception:
        return None


def compute_jitter(arrival_times: list) -> float:
    if len(arrival_times) < 2:
        return 0.0
    jitter = 0.0
    for i in range(1, len(arrival_times)):
        d = abs(arrival_times[i] - arrival_times[i - 1])
        jitter += (d - jitter) / 16.0
    return jitter


def extract_features(pcap_path: str, label: int) -> list:
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"PCAP 읽기 실패: {e}")
        return []

    sessions = {}
    for pkt in packets:
        if not pkt.haslayer(UDP):
            continue
        payload = bytes(pkt[UDP].payload)
        rtp = parse_rtp_header(payload)
        if rtp is None:
            continue
        ssrc = rtp['ssrc']
        if ssrc not in sessions:
            sessions[ssrc] = []
        sessions[ssrc].append({
            'time': float(pkt.time),
            'seq': rtp['seq'],
            'timestamp': rtp['timestamp'],
        })

    if not sessions:
        print(f"  RTP 세션 없음: {pcap_path}")
        return []

    # SSRC가 1개인 경우 패킷 수 기반으로 세션 분리 (SIPp 연속 발송 대응)
    if len(sessions) == 1:
        ssrc, all_pkts = list(sessions.items())[0]
        all_pkts.sort(key=lambda x: x['time'])
        PKTS_PER_SESSION = 50
        sessions = {}
        for i in range(0, len(all_pkts), PKTS_PER_SESSION):
            chunk = all_pkts[i:i+PKTS_PER_SESSION]
            if len(chunk) >= 5:
                sessions[f"{ssrc}_{i//PKTS_PER_SESSION}"] = chunk

    results = []
    for ssrc, pkts in sessions.items():
        if len(pkts) < 5:
            continue
        pkts.sort(key=lambda x: x['time'])
        arrival_times = [p['time'] for p in pkts]
        seq_numbers = [p['seq'] for p in pkts]

        iats = [arrival_times[i] - arrival_times[i-1] for i in range(1, len(arrival_times))]
        avg_latency = sum(iats) / len(iats) if iats else 0.0
        avg_jitter = compute_jitter(arrival_times)
        mean_iat = avg_latency
        iat_variance = sum((x - mean_iat)**2 for x in iats) / len(iats) if iats else 0.0
        expected = (max(seq_numbers) - min(seq_numbers) + 1) if seq_numbers else 1
        actual = len(seq_numbers)
        packet_loss = max(0.0, (expected - actual) / expected) if expected > 0 else 0.0
        gaps = sum(1 for i in range(1, len(seq_numbers)) if (seq_numbers[i] - seq_numbers[i-1]) % 65536 > 1)
        seq_gap_rate = gaps / len(seq_numbers) if seq_numbers else 0.0

        results.append({
            'session_id': f"{Path(pcap_path).stem}_{ssrc}",
            'avg_latency': round(avg_latency, 6),
            'avg_jitter': round(avg_jitter, 6),
            'iat_variance': round(iat_variance, 8),
            'packet_loss': round(packet_loss, 4),
            'seq_gap_rate': round(seq_gap_rate, 4),
            'packet_count': len(pkts),
            'label': label,
        })
    return results


def save_csv(features: list, output_path: str):
    if not features:
        print("저장할 데이터가 없습니다.")
        return
    fieldnames = ['session_id', 'avg_latency', 'avg_jitter', 'iat_variance',
                  'packet_loss', 'seq_gap_rate', 'packet_count', 'label']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)
    print(f"저장 완료: {output_path} ({len(features)}개 세션)")


def main():
    base_dir = Path(__file__).parent.parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    processed_dir = base_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) == 2:
        pcap_path = sys.argv[1]
        label = 1 if 'fraud' in Path(pcap_path).name.lower() else 0
        features = extract_features(pcap_path, label)
        output = processed_dir / (Path(pcap_path).stem + '_features.csv')
        save_csv(features, str(output))
        return

    pcap_files = list(raw_dir.glob('*.pcap'))
    if not pcap_files:
        print(f"PCAP 파일이 없습니다: {raw_dir}")
        return

    all_features = []
    for pcap_file in pcap_files:
        label = 1 if 'fraud' in pcap_file.name.lower() else 0
        print(f"파싱 중: {pcap_file.name} (label={label})")
        features = extract_features(str(pcap_file), label)
        print(f"  → {len(features)}개 세션 추출")
        all_features.extend(features)

    output_path = processed_dir / 'sessions.csv'
    save_csv(all_features, str(output_path))


if __name__ == '__main__':
    main()
