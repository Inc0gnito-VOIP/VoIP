"""
sender.py v4 — 현실적 데이터 생성기
  Normal: 소량 패킷손실 + 가끔 코덱 재협상 + 세션 겹침
  SIM Box 4타입 균등 25%:
    Obvious: 고지터 + 손실 + 코덱전환
    Stealth: interval 15~25ms (정상 흉내), 행동 피처 정상 위장
    Mixed:   interval 20~35ms, 일부 피처 위장
    Premium: interval 10~20ms, 물리 피처 정상과 겹침
"""

import socket
import struct
import random
import time
import threading
import argparse

RTP_PORT = 5004
DST_IP   = "10.0.0.2"


def make_rtp(seq, ts, pt, ssrc):
    hdr = struct.pack(
        '>BBHII',
        0x80,
        pt & 0x7F,
        seq & 0xFFFF,
        ts & 0xFFFFFFFF,
        ssrc
    )
    return hdr + bytes(random.getrandbits(8) for _ in range(160))


def get_simbox_profile():
    """SIM Box 세션 타입 — 4타입 균등 25%."""
    r = random.random()

    if r < 0.25:
        # ── Obvious (25%): 물리+행동 모두 SIM Box 특성
        interval = 0.035
        return {
            'interval':         interval,
            'jitter_std':       random.uniform(0.018, 0.040),
            'pt_switch':        True,
            'loss_rate':        random.uniform(0.02, 0.05),
            'concurrent_delay': random.uniform(0.0, 0.3),
            'n_packets':        random.randint(40, 80),
            'type':             'obvious'
        }
    elif r < 0.50:
        # ── Stealth (25%): 행동 피처 정상처럼 위장
        interval = random.uniform(0.015, 0.025)
        return {
            'interval':         interval,
            'jitter_std':       random.uniform(0.018, 0.040),
            'pt_switch':        False,
            'loss_rate':        0.0,
            'concurrent_delay': random.uniform(3.5, 5.0),
            'n_packets':        random.randint(70, 110),
            'type':             'stealth'
        }
    elif r < 0.75:
        # ── Mixed (25%): 일부 피처만 위장
        interval = random.uniform(0.020, 0.035)
        return {
            'interval':         interval,
            'jitter_std':       random.uniform(0.018, 0.040),
            'pt_switch':        random.choice([True, False]),
            'loss_rate':        random.uniform(0.0, 0.02),
            'concurrent_delay': random.uniform(0.5, 3.0),
            'n_packets':        random.randint(50, 95),
            'type':             'mixed'
        }
    else:
        # ── Premium (25%): 저지연 경로 — 물리 피처 정상 구간과 겹침
        interval = random.uniform(0.010, 0.020)
        return {
            'interval':         interval,
            'jitter_std':       random.uniform(0.006, 0.015),
            'pt_switch':        False,
            'loss_rate':        random.uniform(0.0, 0.01),
            'concurrent_delay': random.uniform(1.0, 3.0),
            'n_packets':        random.randint(60, 100),
            'type':             'premium'
        }


def send_session(profile, ssrc, call_id):
    """단일 RTP 세션 송신."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    mode       = profile['mode']
    n_packets  = profile['n_packets']
    jitter_std = profile['jitter_std']
    loss_rate  = profile['loss_rate']
    pt_switch  = profile.get('pt_switch', False)
    interval   = profile['interval']
    ts_step    = max(1, int(interval * 8000))

    seq = random.randint(0, 1000)
    ts  = random.randint(0, 100000)
    sent = 0

    for i in range(n_packets):
        # codec mismatch: 세션 중간 이후 PT 변경
        if pt_switch and i >= n_packets // 2:
            pt = random.choice([8, 3])   # PCMA or GSM
        else:
            pt = 0

        # packet loss
        if random.random() < loss_rate:
            seq += 1
            ts  += ts_step
            continue

        pkt = make_rtp(seq, ts, pt, ssrc)
        try:
            sock.sendto(pkt, (DST_IP, RTP_PORT))
            sent += 1
        except Exception:
            pass

        seq += 1
        ts  += ts_step

        wait = random.gauss(interval, jitter_std)
        time.sleep(max(wait, 0.001))

    sock.close()
    print(f"[SND] call_id={call_id} mode={mode} type={profile.get('type','normal')} "
          f"ssrc=0x{ssrc:08x} sent={sent}")


def run(mode, n_sessions, n_packets):
    threads = []

    for i in range(n_sessions):
        ssrc = random.randint(0, 0xFFFFFFFF)

        if mode == 'normal':
            profile = {
                'mode':             'normal',
                'interval':         0.010,
                'n_packets':        n_packets,
                'jitter_std':       random.uniform(0.003, 0.006),
                'loss_rate':        random.uniform(0.0, 0.005),
                'pt_switch':        random.random() < 0.02,
                'concurrent_delay': 0,
            }
            delay = i * 1.0

        else:  # simbox
            profile = get_simbox_profile()
            profile['mode'] = 'simbox'
            delay = profile['concurrent_delay']

        def launch(d=delay, p=profile, s=ssrc, c=i):
            time.sleep(d)
            send_session(p, s, c)

        t = threading.Thread(target=launch)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print(f"\n[완료] mode={mode} sessions={n_sessions}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RTP 송신기 v4')
    parser.add_argument('--mode',     choices=['normal', 'simbox'], default='normal')
    parser.add_argument('--sessions', type=int, default=4)
    parser.add_argument('--packets',  type=int, default=60)
    args = parser.parse_args()

    if args.mode == 'normal':
        est = args.sessions * 1
        print(f"[INFO] normal 모드 — 1초 간격, 예상 {est}초 ({est//60}분 {est%60}초)")
    else:
        print("[INFO] simbox 모드 — Obvious/Stealth/Mixed/Premium 각 25%")

    run(args.mode, args.sessions, args.packets)
