# 05. FE 계약: 녹화 업로드 · STT · LLM 리포트 · 영상 타임스탬프

- 작성일: 2026-09-07 · 작성자: 박기호 (BE)
- 상태: **계약 확정본 (BE 결정).** FE는 이 문서대로 붙이면 되고, 불편한 점은 구현 전에 알려 주세요.
- 원칙: **기존 API·WebSocket은 바꾸지 않고 추가만** 합니다. 이미 붙인 면접·리포트 화면은 그대로 동작합니다.

## 0. 한눈에
```
면접 중   FE: MediaRecorder(오디오+비디오 webm) → 5초마다 조각 업로드  POST /recording/chunks
종료(end) BE: 조각 합치기 → 수치 리포트 즉시 생성(지금과 동일) → 백그라운드로 STT → LLM
리포트    FE: GET /report 를 3초마다 다시 조회, status 가 done 이 될 때까지. 영상은 GET /recording 으로 재생
```
구현 순서(BE): ① 녹화 업로드 → ② STT → ③ LLM 리포트 → ④ 영상 재생 URL. FE 는 ①과 ④만 새 작업이고 ②③은 리포트 필드 추가입니다.

## 1. 녹화와 업로드 (FE 작업 큼)

### 1.1 녹화
면접 화면에서 웹캠 스트림 하나로 **오디오+비디오를 같이** 녹화합니다. 프레임 전송(기존 WebSocket)과는 별개입니다.
```ts
const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true });
videoEl.srcObject = stream;                        // 화면 표시 + 기존 프레임 캡처에 그대로 사용
const rec = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp8,opus', videoBitsPerSecond: 800_000 });
let seq = 0;
rec.ondataavailable = async (e) => {
  if (e.data.size === 0) return;
  const fd = new FormData();
  fd.append('chunk', e.data, `${seq}.webm`);
  await apiFetchRaw(`/api/sessions/${sessionId}/recording/chunks?seq=${seq++}`, { method: 'POST', body: fd });
};
rec.start(5000);                                   // 5초마다 ondataavailable
// 면접 종료(end 보내기 직전): rec.stop()  → 마지막 조각까지 업로드된 뒤 end 전송
```
- `mimeType` 은 Chrome/Edge 기준. 지원 안 하면 `MediaRecorder.isTypeSupported` 로 `video/webm` 폴백.
- 비트레이트 800kbps → 5분에 약 30MB. 조각 하나 약 500KB.
- `apiFetchRaw` 는 기존 `apiFetch` 에서 `Content-Type: application/json` 을 빼고 FormData 를 그대로 보내는 버전. (JSON 헤더를 붙이면 multipart 가 깨집니다.)
- **마이크 권한 거부 시**: 영상만 녹화(`audio: false`)하고 계속 진행. STT 는 비어 있는 채로 리포트가 나옵니다.
- 재연결(`resume`) 시 `seq` 는 이어서 증가시키면 됩니다. 서버가 순서대로 붙입니다.

### 1.2 API
| 메서드 | 경로 | 요청 | 응답 |
|---|---|---|---|
| POST | `/api/sessions/{id}/recording/chunks?seq=N` | multipart `chunk` (webm 조각) | `204`. 같은 seq 재전송은 덮어씀(멱등) |
| GET | `/api/sessions/{id}/recording` | | webm 스트리밍 (`Range` 지원, `<video src>` 에 바로 사용). 없으면 `404` |

조각은 세션 상태가 `running` 일 때만 받습니다. 종료 후 5초 안에 도착한 마지막 조각까지는 허용합니다.

## 2. 리포트 확장 (FE 작업 작음)

`GET /api/sessions/{id}/report` 응답에 아래가 **추가**됩니다. 기존 필드는 그대로입니다.
```json
{
  "...기존 필드 그대로...": "",
  "status": { "metrics": "done", "stt": "pending", "llm": "pending", "recording": "done" },
  "recording": { "url": "/api/sessions/{id}/recording", "duration_ms": 312000 },
  "transcript": [
    { "question_index": 0, "text": "안녕하세요, 저는 ...", "words": 142, "speech_ms": 48000 }
  ],
  "llm": {
    "summary": "전체 면접에 대한 3~4문장 총평",
    "per_question": [ { "question_index": 0, "feedback": "답변 내용과 행동을 함께 본 피드백" } ],
    "strengths": ["..."], "improvements": ["..."],
    "model": "openai/gpt-4o-mini"
  }
}
```
| 필드 | 값 | FE 처리 |
|---|---|---|
| `status.*` | `pending` \| `running` \| `done` \| `failed` \| `skipped` | 하나라도 `pending`/`running` 이면 **3초 후 다시 GET**. `skipped` 는 마이크 없음 등으로 건너뜀 |
| `transcript` | STT 완료 전엔 `[]` | 질문별 답변 텍스트 표시 |
| `llm` | 완료 전엔 `null` | 총평·질문별 피드백·강점·개선점 표시. 기존 규칙 기반 `feedback` 은 그대로 남아 있으니 `llm` 이 `null` 일 때의 대체로 사용 |
| `recording` | 녹화 없으면 `null` | 영상 플레이어 `src`. `status.recording` 이 `done` 이어야 재생 가능 |

STT 는 5분 면접 기준 약 20~40초, LLM 은 약 10~30초 걸립니다. 리포트 화면에 "답변 분석 중…" 표시를 두면 됩니다.

## 3. 영상 타임스탬프 (FE 작업 중간)
리포트의 `timeline[].ts_ms` 는 **세션 시작 기준 경과 밀리초**이고, 녹화도 세션 시작 직후부터이므로 그대로 영상 시각입니다.
```ts
videoEl.currentTime = event.ts_ms / 1000;  videoEl.play();
```
- 타임라인 이벤트를 클릭하면 그 시각으로 이동. 서버 추가 작업 없음.
- 녹화 시작이 `start` 보다 1~2초 늦을 수 있어 최대 그 정도 오차가 있습니다. 필요하면 `recording.offset_ms` 를 추가하겠습니다(현재는 0).

## 4. FE 에 부탁하는 순서
1. 1.1 녹화·업로드 (BE ①이 올라가면 바로 붙일 수 있음. 그 전엔 `chunks` 가 `404`)
2. 리포트 화면에 `status` 폴링과 "분석 중" 표시
3. `transcript`, `llm` 렌더링
4. 영상 플레이어 + 타임라인 클릭 이동

## 5. BE 쪽 구현 메모 (참고)
- 저장: `backend/media/{session_id}/` 에 조각 저장 후 종료 시 하나로 합침. 배포 안 하므로 로컬 디스크. git 제외.
- STT: 추론 서버에 faster-whisper **small** 모델(GPU, 한국어) 추가, `POST /v1/transcribe`. BE 가 ffmpeg 로 질문 구간(`session_questions.started_at~ended_at`)별 오디오를 잘라 전송. (팀 결정 2026-09-07)
- LLM: **OpenAI API** (`gpt-4o-mini` 기본, `OPENAI_MODEL` 로 변경). 입력은 수치 리포트 + 질문·답변 텍스트, 출력은 위 `llm` 구조(JSON 강제). 키는 `backend/.env` 의 `OPENAI_API_KEY`, 절대 커밋하지 않음. (팀 결정 2026-09-07)
- 백그라운드 작업: FastAPI `BackgroundTasks`. 상태는 `reports.summary.status` 에 기록.
