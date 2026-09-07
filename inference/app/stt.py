"""faster-whisper 래퍼. 오디오 바이트(webm/wav/mp3 등 ffmpeg 가 읽는 형식) → 텍스트와 구간."""

import io
import logging
import time

from faster_whisper import WhisperModel

log = logging.getLogger("inference.stt")


class SttModel:
    def __init__(self, size: str, device: str, language: str):
        compute = "float16" if device == "cuda" else "int8"
        t = time.perf_counter()
        self.model = WhisperModel(size, device=device, compute_type=compute)
        self.language = language
        self.name = f"whisper-{size}"
        log.info("whisper %s loaded on %s (%s) in %.1fs", size, device, compute, time.perf_counter() - t)

    def transcribe(self, audio: bytes, language: str | None = None) -> dict:
        segments, info = self.model.transcribe(
            io.BytesIO(audio),
            language=language or self.language,
            beam_size=5,
            vad_filter=True,  # 무음 구간 제거 → 환각 감소, 속도 향상
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # 같은 구절 반복 생성(환각 루프) 억제
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )
        segs = []
        for seg in segments:
            text = seg.text.strip()
            if not text or (segs and text == segs[-1]["text"]):  # 연속 중복 구절 제거
                continue
            segs.append({"start_ms": int(seg.start * 1000), "end_ms": int(seg.end * 1000), "text": text})
        text = " ".join(s["text"] for s in segs).strip()
        duration_ms = int(info.duration * 1000)
        speech_ms = min(sum(s["end_ms"] - s["start_ms"] for s in segs), duration_ms)  # VAD 패딩으로 합이 길이를 넘을 수 있어 상한
        return {
            "text": text,
            "segments": segs,
            "language": info.language,
            "duration_ms": duration_ms,
            "speech_ms": speech_ms,
        }
