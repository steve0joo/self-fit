"""실시간 조건(초당 N장)으로 프레임을 보내며 타임스탬프·왕복 시간·서버 단계별 시간·결과를 한 줄씩 출력한다.

사용 예)
  python3 scripts/live_bench.py samples/images.jpg "samples/images (1).jpg"         # 이미지들을 번갈아 3fps 로 20초
  python3 scripts/live_bench.py --fps 5 --seconds 10 samples/images.jpg
  python3 scripts/live_bench.py --video my.mp4                                       # opencv 가 있을 때 (pip install opencv-python-headless)
  python3 scripts/live_bench.py --jsonl run.jsonl samples/images.jpg                  # 결과를 JSON Lines 로 저장

표준 라이브러리만 사용. 서버 기본 주소 http://localhost:9000, 토큰은 INFERENCE_TOKEN 환경변수.
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import unicodedata
from datetime import datetime


def _w(text: str) -> int:
    """터미널 표시 폭. 한글 등 동아시아 전각 문자는 2칸."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def pad(text, width: int, align: str = "<") -> str:
    """표시 폭 기준으로 정렬. align '<' 왼쪽, '>' 오른쪽."""
    text = str(text)
    gap = max(0, width - _w(text))
    return text + " " * gap if align == "<" else " " * gap + text


# (헤더, 폭, 정렬)
COLS = [("시각", 12, "<"), ("#", 4, ">"), ("간격", 5, ">"), ("왕복", 5, ">"), ("서버", 5, ">"),
        ("검출", 4, ">"), ("시선", 4, ">"), ("감정", 4, ">"), ("집중", 4, ">"), ("|", 1, "<"),
        ("얼굴", 4, "<"), ("yaw", 6, ">"), ("pitch", 6, ">"), ("conf", 5, ">"), ("감정top", 7, "<"), ("집중top", 8, "<"), ("입력", 0, "<")]


def row(values) -> str:
    return " ".join(pad(v, w, a) for v, (_, w, a) in zip(values, COLS, strict=True))


def header() -> str:
    return row([h for h, _, _ in COLS])


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="*", help="JPEG/PNG 파일들. 여러 개면 번갈아 전송")
    p.add_argument("--video", help="동영상 파일. 프레임을 뽑아 전송 (opencv 필요)")
    p.add_argument("--url", default=os.environ.get("INFERENCE_URL", "http://localhost:9000"))
    p.add_argument("--fps", type=float, default=3.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--session", default=f"bench-{int(time.time())}")
    p.add_argument("--max-side", type=int, default=640, help="동영상 프레임 긴 변 크기 (FE 는 224)")
    p.add_argument("--jsonl", help="프레임별 결과를 저장할 파일")
    return p.parse_args()


def frames_from_images(paths):
    blobs = [(os.path.basename(x), open(x, "rb").read()) for x in paths]
    i = 0
    while True:
        yield blobs[i % len(blobs)]
        i += 1


def frames_from_video(path, max_side):
    try:
        import cv2
    except ImportError:
        sys.exit("동영상 입력에는 opencv 가 필요합니다: pip install opencv-python-headless")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"동영상을 열 수 없습니다: {path}")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        h, w = frame.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        n += 1
        yield f"frame{n}", buf.tobytes()


def post(url, session, blob, token):
    req = urllib.request.Request(
        f"{url}/v1/analyze?session_id={session}", data=blob, method="POST",
        headers={"Content-Type": "image/jpeg", **({"X-Inference-Token": token} if token else {})},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main():
    a = parse()
    if not a.images and not a.video:
        sys.exit("이미지 파일 또는 --video 를 지정하세요. (예: python3 scripts/live_bench.py samples/images.jpg)")
    token = os.environ.get("INFERENCE_TOKEN", "")
    try:
        with urllib.request.urlopen(f"{a.url}/v1/health", timeout=10) as r:
            print("서버:", r.read().decode())
    except urllib.error.URLError as e:
        sys.exit(f"추론 서버에 연결할 수 없습니다 ({a.url}): {e}")

    src = frames_from_video(a.video, a.max_side) if a.video else frames_from_images(a.images)
    interval = 1.0 / a.fps
    out = open(a.jsonl, "w") if a.jsonl else None
    rtts, server_totals, faces, n_fail = [], [], 0, 0
    print(f"세션 {a.session} · {a.fps:g}fps · {a.seconds:g}초 · 서버 {a.url}")
    print("단위 ms. 간격=이전 프레임 전송 후 경과(목표 {:.0f}). 집중 열에 * 표시된 프레임에서만 집중 모델이 실제로 실행됨.\n".format(1000 / a.fps))
    print(header())
    print("-" * _w(header()))
    t_end = time.perf_counter() + a.seconds
    n, t_prev = 0, None
    while time.perf_counter() < t_end:
        t_next = time.perf_counter() + interval
        try:
            name, blob = next(src)
        except StopIteration:
            break
        n += 1
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        t0 = time.perf_counter()
        gap = f"{(t0 - t_prev) * 1000:.0f}" if t_prev else "-"
        t_prev = t0
        try:
            res = post(a.url, a.session, blob, token)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"{ts:<12} {n:>4}  실패: {e}")
            continue
        rtt = (time.perf_counter() - t0) * 1000
        tm = res.get("timing_ms", {})
        srv = sum(tm.values())
        rtts.append(rtt); server_totals.append(srv)
        g = res.get("gaze") or {}
        emo = (res.get("emotion") or {}).get("top", "-")
        att = (res.get("attention") or {}).get("top", "-")
        att_ran = tm.get("attention", 0) > 0
        if res.get("face_found"):
            faces += 1
        if n > 1 and (n - 1) % 10 == 0:
            print("-" * _w(header()))
        print(row([ts, n, gap, f"{rtt:.0f}", srv, tm.get("detect", "-"), tm.get("gaze", "-"), tm.get("emotion", "-"),
                   (f"{tm['attention']}*" if att_ran else "0"), "|",
                   "O" if res.get("face_found") else "X",
                   f"{g.get('yaw_deg', float('nan')):.1f}", f"{g.get('pitch_deg', float('nan')):.1f}", f"{g.get('confidence', 0):.2f}",
                   emo, att, name]))
        if out:
            out.write(json.dumps({"ts": ts, "n": n, "gap_ms": gap, "rtt_ms": round(rtt, 1), "input": name, **res}, ensure_ascii=False) + "\n")
        sleep = t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
    if out:
        out.close()

    if not rtts:
        sys.exit("성공한 요청이 없습니다.")
    srt = sorted(rtts)
    p50, p95 = srt[len(srt) // 2], srt[min(len(srt) - 1, int(len(srt) * 0.95))]
    print("\n요약")
    print(f"  프레임 {n}장 (성공 {len(rtts)}, 실패 {n_fail}), 얼굴 검출 {faces}/{len(rtts)}")
    print(f"  왕복  평균 {statistics.mean(rtts):.0f}ms · p50 {p50:.0f}ms · p95 {p95:.0f}ms · 최대 {max(rtts):.0f}ms")
    print(f"  서버  평균 {statistics.mean(server_totals):.0f}ms (검출+시선+감정+집중), 나머지는 전송·직렬화")
    budget = 1000 / a.fps
    print(f"  {a.fps:g}fps 예산 {budget:.0f}ms 대비 p95 {'여유' if p95 < budget else '초과'} ({p95 / budget * 100:.0f}%)")
    if a.jsonl:
        print(f"  저장: {a.jsonl}")


if __name__ == "__main__":
    main()
