"""
test_sender.py v2 — Hard Examples 테스트 데이터 생성기
  Normal 700세션: 소량 손실 + 가끔 코덱 재협상 (학습과 동일)
  SIM Box 300세션 (Hard Examples 비율):
    Obvious  30 (10%) — 쉬운 케이스 최소화
    Stealth  90 (30%) — 정상 흉내
    Mixed    90 (30%) — 일부 위장
    Premium  90 (30%) — 물리 피처 정상과 겹침

사용법:
  [SIM Box] sudo ip netns exec sender python3 /home/grute/test_sender.py --packets 60
  [정상]   sudo ip netns exec sender python3 /home/grute/sender.py --mode normal --sessions 700 --packets 60
"""

import socket
import struct
import random
import time
import threading
import argparse

RTP_PORT = 5004
DST_IP   = "10.0.0.2"

# ── Hard Examples 비율 ──
TYPE_COUNTS = {
    'obvious':   30,
    'stealth':   90,
    'mixed':     90,
    'premium':   90,
}
N_SIMBOX = sum(TYPE_COUNTS.values())  # 300


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


def get_simbox_profile_by_type(sim_type):
    """지정된 타입의 SIM Box 프로필 반환."""
    if sim_type == 'obvious':
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
    elif sim_type == 'stealth':
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
    elif sim_type == 'mixed':
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
    else:  # premium — 테스트에서 더 넓은 범위
        interval = random.uniform(0.010, 0.022)
        return {
            'interval':         interval,
            'jitter_std':       random.uniform(0.004, 0.012),
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
        if pt_switch and i >= n_packets // 2:
            pt = random.choice([8, 3])
        else:
            pt = 0

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


def run(n_packets):
    print(f"[INFO] 테스트 SIM Box 데이터 생성 — Hard Examples 비율")
    print(f"  SIM Box: {N_SIMBOX}세션")
    for t, c in TYPE_COUNTS.items():
        print(f"    {t}: {c}세션 ({c/N_SIMBOX*100:.0f}%)")
    print()

    threads = []
    call_id = 0

    for sim_type, count in TYPE_COUNTS.items():
        for _ in range(count):
            ssrc = random.randint(0, 0xFFFFFFFF)
            profile = get_simbox_profile_by_type(sim_type)
            profile['mode'] = 'simbox'
            profile['n_packets'] = n_packets if n_packets else profile['n_packets']
            delay = profile['concurrent_delay']

            def launch(d=delay, p=profile, s=ssrc, c=call_id):
                time.sleep(d)
                send_session(p, s, c)

            t = threading.Thread(target=launch)
            t.start()
            threads.append(t)
            call_id += 1

    for t in threads:
        t.join()
    print(f"\n[완료] SIM Box {N_SIMBOX}세션 (Obvious {TYPE_COUNTS['obvious']} / "
          f"Stealth {TYPE_COUNTS['stealth']} / Mixed {TYPE_COUNTS['mixed']} / "
          f"Premium {TYPE_COUNTS['premium']})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='테스트 데이터 생성기 (Hard Examples)')
    parser.add_argument('--packets', type=int, default=60)
    args = parser.parse_args()

    run(args.packets)
