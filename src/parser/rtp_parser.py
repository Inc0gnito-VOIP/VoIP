"""
rtp_parser.py v2 — PCAP → 6피처 CSV 추출
  피처: avg_latency, avg_jitter, iat_variance, packet_loss, seq_gap_rate, codec_mismatch
  제거: concurrent_calls (캡처 부산물), call_pattern (avg_latency 중복)
"""

import struct, math, csv, argparse
from scapy.all import rdpcap, UDP, Raw

FEATURES = [
    "session_id", "avg_latency", "avg_jitter", "iat_variance",
    "packet_loss", "seq_gap_rate", "codec_mismatch", "label"
]

def parse_rtp_packet(pkt):
    if not (UDP in pkt and Raw in pkt):
        return None
    raw = bytes(pkt[Raw])
    if len(raw) < 12:
        return None
    version = (raw[0] >> 6) & 0x3
    if version != 2:
        return None
    pt   = raw[1] & 0x7F
    seq  = int.from_bytes(raw[2:4], 'big')
    ts   = int.from_bytes(raw[4:8], 'big')
    ssrc = int.from_bytes(raw[8:12], 'big')
    return {"pt": pt, "seq": seq, "ts": ts, "ssrc": ssrc, "time": float(pkt.time)}

def extract_sessions(pcap_path):
    pkts = rdpcap(pcap_path)
    sessions = {}
    for pkt in pkts:
        r = parse_rtp_packet(pkt)
        if r is None:
            continue
        sessions.setdefault(r["ssrc"], []).append(r)
    for ssrc in sessions:
        sessions[ssrc].sort(key=lambda x: x["time"])
    return sessions

def calc_jitter_rfc3550(arrivals, timestamps):
    if len(arrivals) < 2:
        return 0.0
    jitter = 0.0
    for i in range(1, len(arrivals)):
        d = abs(
            (arrivals[i] - arrivals[i-1]) -
            (timestamps[i] - timestamps[i-1]) / 8000.0
        )
        jitter += (d - jitter) / 16.0
    return jitter

def compute_features(ssrc, pkts):
    if len(pkts) < 2:
        return None
    times = [p["time"] for p in pkts]
    seqs  = [p["seq"]  for p in pkts]
    pts   = [p["pt"]   for p in pkts]
    tss   = [p["ts"]   for p in pkts]

    iats = [times[i] - times[i-1] for i in range(1, len(times))]

    avg_latency  = sum(iats) / len(iats)
    avg_jitter   = calc_jitter_rfc3550(times, tss)
    mean_iat     = avg_latency
    iat_variance = sum((x - mean_iat)**2 for x in iats) / len(iats)

    seq_range    = max(seqs) - min(seqs) + 1
    packet_loss  = max(0.0, (seq_range - len(pkts)) / seq_range) if seq_range > 0 else 0.0

    gaps = 0
    for i in range(1, len(seqs)):
        diff = (seqs[i] - seqs[i-1]) & 0xFFFF
        if diff != 1:
            gaps += 1
    seq_gap_rate   = gaps / (len(seqs) - 1) if len(seqs) > 1 else 0.0

    dominant_pt    = max(set(pts), key=pts.count)
    codec_mismatch = sum(1 for p in pts if p != dominant_pt) / len(pts)

    return {
        "avg_latency":     round(avg_latency,    6),
        "avg_jitter":      round(avg_jitter,      6),
        "iat_variance":    round(iat_variance,    8),
        "packet_loss":     round(packet_loss,     4),
        "seq_gap_rate":    round(seq_gap_rate,    4),
        "codec_mismatch":  round(codec_mismatch,  4),
    }

def extract_features(pcap_path, label, output_rows, session_counter):
    sessions = extract_sessions(pcap_path)
    for ssrc, pkts in sessions.items():
        feats = compute_features(ssrc, pkts)
        if feats is None:
            continue
        row = {"session_id": session_counter[0], "label": label}
        row.update(feats)
        output_rows.append(row)
        session_counter[0] += 1
    print(f"  {pcap_path}: {len(sessions)}개 세션 추출")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal",  required=True)
    ap.add_argument("--simbox",  required=True)
    ap.add_argument("--output",  default="sessions.csv")
    args = ap.parse_args()
    rows, counter = [], [0]
    print("정상 세션 추출 중...")
    extract_features(args.normal, 0, rows, counter)
    print("SIM Box 세션 추출 중...")
    extract_features(args.simbox, 1, rows, counter)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n완료: {args.output} ({len(rows)}개 세션)")
    print(f"  정상: {sum(1 for r in rows if r['label']==0)}개 / SIM Box: {sum(1 for r in rows if r['label']==1)}개")
