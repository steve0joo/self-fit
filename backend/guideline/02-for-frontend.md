# 02. FE 인수인계: BE 인터페이스

백엔드가 설계한 API와 WebSocket 규격 중 FE가 붙여야 하는 부분을 정리했습니다. Phase 1 구현이 끝나면 이 문서대로 Mock 모드에서 전체 흐름이 동작하므로, 모델 없이도 FE 개발을 시작할 수 있습니다. 전체 명세는 `../docs/03-phase1-design.md` 5절, 6절.

**현재 상태 (2026-09-06):** 아래 전부 **Mock 모드로 구현되어 있습니다.** 세션 생성 → WebSocket → 토스트 이벤트 → 종료 → 리포트까지 실제 서버로 붙일 수 있습니다. 모델만 가짜이고 규칙적으로 이벤트를 만듭니다(시작 후 약 3초 뒤 시선 이탈, 15초쯤 긴장 표정, 26초쯤 집중 저하, 70초쯤 얼굴 사라짐).

## 1. 준비
```bash
cd frontend && cp .env.local.example .env.local   # 값은 팀 채널에서
npm install && npm run dev
```
BE는 `http://localhost:8000`에서 돌고, `http://localhost:8000/docs`에서 API를 직접 눌러 볼 수 있습니다.

**BE가 내 PC가 아니라 백엔드 담당 노트북에서 돌 때(같은 Wi-Fi):** `.env.local`의 `NEXT_PUBLIC_API_URL`을 `http://<노트북IP>:8000`으로 바꾸면 됩니다. FE는 내 PC에서 `localhost:3000`으로 띄우세요. 웹캠은 `localhost`에서만 브라우저가 허용하기 때문입니다. 노트북 IP는 백엔드 담당에게 물어보세요.
연결이 되는지 보려면 `http://localhost:3000/dev/connect`를 여세요. 헬스체크와 로그인 토큰 검증이 ✅로 보이면 됩니다.

## 2. 로그인과 토큰
- 로그인은 Supabase JS로 합니다. `lib/supabase.ts`의 `getSupabase()`로 클라이언트를 얻습니다.
- BE 주소는 `NEXT_PUBLIC_API_URL`로 넣어 주세요. `.env.local.example`에 이 줄을 추가해 두면 좋습니다: `NEXT_PUBLIC_API_URL=http://localhost:8000`
- BE를 부를 때는 **모든 요청에** 토큰을 붙입니다.

```ts
import { getSupabase } from '@/lib/supabase';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function api(path: string, init: RequestInit = {}) {
  const { data } = await getSupabase().auth.getSession();
  const token = data.session?.access_token;
  return fetch(`${API_URL}${path}`, {
    ...init,
    headers: { ...init.headers, Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
  });
}
```
토큰이 없거나 만료되면 BE가 `401`을 돌려줍니다. 그러면 로그인 화면으로 보내면 됩니다.

## 3. 면접 한 번의 흐름
```
질문 목록 조회 → 세션 생성 → WebSocket 연결 → start → 프레임 전송(3fps) → 질문 전환 → end → 리포트 조회
```

| 단계 | 호출 | 받는 것 |
|---|---|---|
| 질문 목록 | `GET /api/questions` | `[{id, text, sort_order}]` |
| 세션 생성 | `POST /api/sessions` body `{mode:"live"}` | `{id, questions:[...], ws_url}` |
| 실시간 | `WebSocket /ws/sessions/{id}?token=<access_token>` | 아래 4절 |
| 리포트 | `GET /api/sessions/{id}/report` | 아래 5절 |
| 과거 목록 | `GET /api/sessions?limit=20&offset=0` | `{items:[...], total}` |

## 4. WebSocket
브라우저 WebSocket은 헤더를 못 붙이므로 토큰은 쿼리로 보냅니다.

```ts
const ws = new WebSocket(`${API_URL.replace('http', 'ws')}/ws/sessions/${sessionId}?token=${token}`);
ws.binaryType = 'arraybuffer';

ws.onopen = () => ws.send(JSON.stringify({ type: 'start' }));   // ready 를 받은 뒤 보내도 됨

// 프레임: 캔버스에 224px로 그려서 JPEG로 보냄. 초당 3장.
setInterval(() => {
  canvas.getContext('2d')!.drawImage(video, 0, 0, 224, 224);
  canvas.toBlob((blob) => blob && ws.send(blob), 'image/jpeg', 0.7);
}, 333);

// 다음 질문으로 넘어갈 때
ws.send(JSON.stringify({ type: 'question', index: 1 }));

// 면접 끝
ws.send(JSON.stringify({ type: 'end' }));

ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === 'event' && toastEnabled) showToast(msg.icon, msg.message);   // 사용자가 켠 경우만
  if (msg.type === 'report_ready') router.push(`/interview/report?session=${sessionId}`);
};
```

BE가 보내는 메시지:
| type | 언제 | 내용 |
|---|---|---|
| `ready` | 연결 직후 | 인증 성공. `fps_hint: 3`, `status`(created면 start, running이면 resume 보내기) |
| `started` | start/resume 처리 후 | `{status, question_index}`. 이 뒤부터 프레임을 보내면 됨 |
| `event` | 판정 규칙에 걸릴 때 | `{icon, message, event_type, severity}`. 토스트가 **켜져 있으면** 그대로 띄우고, 꺼져 있으면 무시. BE는 설정과 무관하게 항상 보냄 |
| `result` | 프레임마다 | 시선·감정·집중 수치. MVP에서는 무시해도 됨 |
| `question_ack` | 질문 전환 반영 후 | `{index}` |
| `report_ready` | end 처리 후 | 리포트 화면으로 이동 |
| `error` | 복구 가능한 오류 | `{code, message}`. 연결은 유지됨 |

**재연결:** BE가 재시작(`--reload` 포함)하거나 네트워크가 흔들리면 연결이 끊깁니다. `onclose`에서 code가 `1000`(정상 종료)이나 `44xx`(인증·세션 오류)가 아니면 2초 뒤 같은 URL로 다시 연결하고 `start` 대신 `{type:"resume"}`을 보내 주세요. BE가 세션 상태를 DB에서 읽어 이어 갑니다. 3회 실패하면 사용자에게 안내 후 `/finish`를 호출해 리포트라도 남깁니다.

연결이 끊기면 close code로 이유를 알 수 있습니다. `4401` 토큰 오류, `4404` 세션 없음, `4409` 이미 끝난 세션.

토스트 문구는 BE가 내려줍니다. 지금 FE에 하드코딩된 4종과 같은 문구이고, 집중 상태용 "😴 집중이 흐트러진 것 같아요"가 하나 추가됩니다.

## 5. 리포트 JSON
현재 리포트 화면의 항목과 1:1로 대응합니다.
```json
{
  "overview": { "gaze_hold_rate": 0.82, "stable_emotion_rate": 0.74, "attention_rate": 0.81, "event_count": 6 },
  "per_question": [ { "order_index": 0, "text": "...", "gaze_hold_rate": 0.78, "dominant_emotion": "중립", "attention_rate": 0.85, "event_count": 2 } ],
  "feedback": [ { "question_index": 0, "note": "초반 시선 이탈이 2회 감지됐어요. ..." } ],
  "emotion_distribution": { "중립": 0.58, "불안": 0.12, "당황": 0.05, "기쁨": 0.19, "기타": 0.06 },
  "timeline": [ { "ts_ms": 12000, "question_index": 0, "type": "gaze_off", "message": "..." } ]
}
```
- `gaze_hold_rate` → 시선 유지율, `stable_emotion_rate` → 안정 표정 비율, `event_count` → 알림 횟수.
- `attention_rate`(집중 유지율)는 새 지표라 타일 하나를 추가해야 합니다.
- `feedback` 배열 길이는 가변입니다.

## 6. FE에 부탁하는 것
1. 로그인·회원가입 화면과 로그아웃. 면접 시작은 로그인 후에만.
2. 면접 화면의 하드코딩 질문을 `GET /api/questions`로 교체.
3. 프레임 전송(4절 코드)과 토스트를 서버 메시지로 교체. **토스트 on/off 토글**을 면접 화면에 두고 설정은 FE(`localStorage`)에 저장. 꺼도 리포트에는 이벤트가 그대로 남습니다.
4. 리포트 화면을 5절 JSON으로 렌더링.
5. `/dev/connect` 페이지는 BE가 만든 확인용이니 통합 후 지워도 됩니다.

BE가 Phase 1을 끝내면 **Mock 모드**로 위 흐름 전체가 돕니다. 모델 없이도 토스트와 리포트가 나오니 FE 개발을 먼저 시작할 수 있습니다.
