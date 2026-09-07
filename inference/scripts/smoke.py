"""실행 중인 추론 서버에 샘플 이미지를 보내 결과와 시간을 출력한다.
사용: python scripts/smoke.py [URL] [이미지...]   기본 URL http://localhost:9000, 기본 이미지 samples/*.jpg"""

import glob
import json
import os
import sys
import time
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9000"
images = sys.argv[2:] or sorted(glob.glob("samples/*.jpg"))
token = os.environ.get("INFERENCE_TOKEN", "")
hdr = {"Content-Type": "image/jpeg", **({"X-Inference-Token": token} if token else {})}

with urllib.request.urlopen(f"{url}/v1/health", timeout=10) as r:
    print("health:", r.read().decode())
for path in images:
    data = open(path, "rb").read()
    for i in range(3):  # 3회 보내 지연 안정치 확인
        t = time.perf_counter()
        req = urllib.request.Request(f"{url}/v1/analyze?session_id=smoke", data=data, headers=hdr, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            out = json.loads(r.read())
        ms = int((time.perf_counter() - t) * 1000)
        if i == 2:
            print(f"{os.path.basename(path)}: {ms}ms 왕복 | face={out['face_found']} gaze={out.get('gaze')} emotion_top={(out.get('emotion') or {}).get('top')} timing={out['timing_ms']}")
