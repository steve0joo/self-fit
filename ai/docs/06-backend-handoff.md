# 06. 백엔드 납품 (스펙 §7 체크리스트)

- 작성일: 2026-09-07
- 브랜치: `feature/cv-preprocessing-model`
- 상태: **납품물 확정. 서버 반영은 `inference/` PR로 진행, K-of-N은 백엔드 별도 이슈**
- 선행: [`04-service-calibration.md`](./04-service-calibration.md), [`05-model-comparison.md`](./05-model-comparison.md), 스펙 [`emotion-finetune-spec.md`](./emotion-finetune-spec.md) §7
- 대상 코드: `inference/app/`, `backend/app/analysis/`

## 1. 먼저 알아야 할 것

**MediaPipe 파라미터는 이미 일치한다.** 스펙 §7.1 Q1로 물어보려던 `model_selection=0` /
`min_detection_confidence=0.5`는 서버([`detector.py`](../../inference/app/detector.py))와 학습
([`preprocess.py` §85-86](../scripts/preprocess.py))이 같은 값을 쓰고 있다. Q2(RGB/BGR)도 서버가
MediaPipe에 RGB를 넣고 있어 문제없다. **이 두 질문은 닫힌다.**

실제로 갈라져 있는 것은 그 상수가 아니라 **박스를 어떻게 쓰느냐**와 **출력을 어떻게 판정하느냐**다.

| 구분 | 항목 | 조치 |
|---|---|---|
| A. 이미 일치 | MediaPipe 파라미터, 채널 순서, FE 프레임 변환 | 확인만 |
| B. 서버가 학습과 다름 | 얼굴 선택 규칙, 박스 정사각형화, 48px 보간 | `inference/` 수정 |
| C. 새로 전달 | 가중치·4클래스 아키텍처, bias·τ | `inference/` 수정 |
| D. 별도 이슈 | K-of-N 판정 | `backend/` 수정 |

---

## 2. 납품물과 전달 방법

스펙 §7의 5개 항목을 **네 개 채널**로 나눠 전달한다. 문서로 수치를 옮겨 적는 채널은 하나도 없다 —
옮겨 적는 순간 재튜닝 때 조용히 어긋나기 때문이다.

| # | 납품물 | 채널 | 비고 |
|---|---|---|---|
| 1 | 가중치 `emotionnet_v2.pth` | **GitHub Release 또는 Drive** | git은 `*.pth`를 무시한다(루트 `.gitignore:16`). 18,964,464 bytes / sha256 `4f5a18fe0b107aaa545bedeedb588de085b064dd9eaa76e3c872e2ebf502209c` |
| 2 | 아키텍처 | **코드** — `inference/app/models/emotionnet.py` 그대로 | 백본 무수정. `fc3`만 7→4 출력. 서버는 `num_classes=4`로 생성만 바꾸면 된다 |
| 3 | 전처리 스펙 | **코드** — `inference/` PR | §3 |
| 4 | 출력 스펙 (클래스·bias·τ) | **`meta.json`을 서버가 읽는다** | §4. 하드코딩 금지 |
| 5 | 성능표 | **문서** — §7, [`04` §5.2](./04-service-calibration.md) | Set B는 여전히 비어 있음 |

**체크포인트 형식은 `{'model': state_dict}`** ([`train.py` §581](../scripts/train.py))이고, 서버의
기존 로더(`_load(weights)["model"]`, [`predictors.py`](../../inference/app/predictors.py))와 그대로 호환된다.

**`meta.json`은 가중치와 같은 디렉터리에 함께 배포한다.** `emotionnet_v2.pth` 옆에
`emotionnet_v2.meta.json`으로 두고 서버가 기동 시 읽는다. bias와 τ는 체크포인트마다 다시 튜닝되는
값이고([`04` §5](./04-service-calibration.md) — 시드마다 최적 bias가 크게 다르다), 코드에 박아 두면
가중치만 교체했을 때 잘못된 조합이 조용히 돈다.

---

## 3. 서버가 학습과 다른 지점 3개

세 가지 모두 감정 모델의 입력 픽셀을 바꾼다. 스펙의 비협상 항목(0% 마진 크롭)에 걸린 부분이다.

### 3.1 다중 얼굴 선택 규칙 — 면적 최대 → 검출 점수 최고

- 학습: `max(det.detections, key=lambda d: d.score[0])` ([`preprocess.py` §259](../scripts/preprocess.py), 스펙 §2.3)
- 서버: 면적 최대 (`detector.py`의 `best_area` 루프)

배경에 사람이 잡히는 상황은 가정이 아니라 관측된 사실이다 —
[`04` §7.1](./04-service-calibration.md)의 제출 이미지 3장 모두 뒤에 2~3명이 있었고, 그때 점수 규칙은
피사체(0.885)를 배경(0.621)보다 높게 봤다. 면접자가 화면에서 가장 크다는 보장은 없다(뒤에 선 사람이
카메라에 더 가까울 수 있다). **점수 규칙으로 통일한다.**

### 3.2 박스 정사각형화·패딩 — 제거

- 학습: MediaPipe 원본 박스를 그대로, 화면 밖은 **클리핑**(패딩 없음). 0% 마진
- 서버: 긴 변 기준 정사각 보정 후, 화면 밖은 **검정 패딩**

정사각 보정은 박스를 세로로 늘려(MediaPipe 박스의 종횡비 중앙값은 0.73, 스펙 Appendix A) 학습이 한 번도
본 적 없는 영역(이마 위·턱 아래)을 끌어들인다. 검정 패딩은 더 나쁘다 — 학습 데이터에 검은 띠가 붙은
얼굴은 없다.

**단, 이 수정은 감정 경로에만 적용한다.** 시선(L2CS, 마진 0.20)과 집중(Former-DFER)은 정사각 크롭을
전제로 검증된 모델이라 기존 동작을 유지해야 한다. 검출기는 **원본 박스를 돌려주고**, 정사각형이 필요한
경로만 `Face.squared()`로 변환한다.

### 3.3 48×48 보간 — `INTER_AREA` → `INTER_LINEAR`

- 계약: bilinear 스트레치 ([`train.py` §90-92](../scripts/train.py) — 원본 `train.py`의 `Resize((48,48))`이 bilinear이라 맞춘 것). `meta.json`의 `input.resize`
- 서버: `INTER_AREA`

**160px 중간 단계를 넣지 말 것.** 학습 캐시가 160px인 것은 열화 증강에 헤드룸이 필요해서지 계약이
아니다. val/test 경로는 `160 → 100px + JPEG q75 → 48 bilinear`이고, 서빙 크롭은 224 프레임 안에서
이미 ~84px(p5/중앙값/p95 = 69/84/110, [`04` §3.2](./04-service-calibration.md))이며 FE가 JPEG를 이미
걸었다. 즉 **서빙에서는 크롭 → 48 bilinear 한 번**이 학습의 마지막 리사이즈에 대응한다.

### 3.4 그레이스케일은 손대지 말 것

서버의 `cv2.COLOR_BGR2GRAY`(BGR 배열)와 학습의 `cv2.COLOR_RGB2GRAY`(RGB 배열)는 **같은 연산이다** —
각자 올바른 채널에 올바른 가중치를 건다. 스펙 §2.4가 경고한 것은 "RGB 배열에 BGR2GRAY를 거는" 경우이고,
서버는 거기 해당하지 않는다. 통일한다고 잘못 바꾸면 그때 오차(평균 3.8 / 최대 48)가 생긴다.

---

## 4. 모델 계약 — 클래스·bias·τ

### 4.1 클래스

```
["기쁨", "당황", "불안", "중립"]     # happy, embarrassed, anxious, neutral
```

원본 7클래스(`분노·상처·슬픔` 포함)가 아니다. 백엔드 [`types.py`](../../backend/app/analysis/types.py)의
`EMOTION_LABELS`도 함께 바뀐다. `EMOTION_USED`(서비스 4종)는 이미 이 넷이라 `'기타'` 분기는 사실상
죽는다 — 남겨 두어도 무해하다.

### 4.2 판정 순서 (순서를 바꾸면 틀린다)

```python
logprob  = model(x)                      # 출력은 LogSoftmax (원본과 동일)
adjusted = logprob + bias                # bias = [0.0, 1.4, 0.0, 1.9]
prob     = softmax(adjusted)
top      = argmax(prob)
accepted = prob.max() >= tau             # tau = 0.98
```

- **bias는 argmax 이전, τ 비교 이전에 더한다.** softmax 이후에 더하는 것은 틀린다.
- bias 순서는 클래스 순서와 같다: `[기쁨, 당황, 불안, 중립]`.
- τ는 **bias 적용 후 확률 기준**이다. `meta.json`의 `tau_biased`(0.98)를 쓴다. `tau`(bias 없는 척도에서
  고른 값)와 혼동하지 말 것.
- 현재 서버는 `torch.exp()`만 하고 bias가 없다. `exp(logprob)`와 `softmax(logprob)`는 같지만,
  bias가 들어가면 더는 같지 않다.

### 4.3 bias가 왜 필요한가

학습은 클래스당 6,800개로 균형을 맞췄으므로(스펙 §1.3) 모델은 **균등 prior**를 학습했다. 실제 면접
스트림은 중립이 지배적이다. bias 없이 서비스에 올리면 **중립 프레임의 16.8%를 불안으로 부른다**(bias
적용 시 7.65%). 이건 모델 결함이 아니라 prior 불일치이고, 재학습 없이 상수 덧셈으로 교정된다.

가정 하나가 딸려 있다. 서비스 prior(중립 .60 / 불안 .15 / 당황 .15 / 기쁨 .10)는 **측정값이 아니라
가정이다** — 실제 면접 스트림의 감정 분포는 아무도 재지 않았다. 다만 **bias가 필요하다는 결론 자체는
prior 값에 둔감하다**(중립이 균등 25%보다 많기만 하면 방향은 같다).

### 4.4 τ를 어떻게 쓸 것인가

τ는 "이 프레임의 판정을 믿을지"를 가르는 값이지 이벤트 임계값이 아니다. test 커버리지 **0.5993**,
통과분 정확도 **0.9466** — 즉 **프레임의 40%는 판정을 버린다.** 추론 서버는 `accepted` 플래그를 응답에
실어 보내고, **버릴지 말지는 백엔드 판정 규칙이 정한다**(§5).

---

## 5. 판정 규칙 — K-of-N (백엔드 별도 이슈)

현재 규칙은 `불안이 top이고 p >= 0.5인 상태가 5초 지속`이다
([`rules.py` §65](../../backend/app/analysis/rules.py), [`config.py` §32-33](../../backend/app/config.py)).
이걸 **최근 N=15프레임 중 K=7프레임**으로 바꾸는 것이 [`04` §6](./04-service-calibration.md)의 결론이다.
FE가 333ms 간격으로 보내므로(3fps) N=15는 기존 5초 창과 같다.

| | 오탐 (독립 가정 하한) | 검출 |
|---|---|---|
| K=6 | 0.0005 | 0.9755 |
| **K=7 (채택)** | **0.0001** | **0.9268** |
| K=8 | 0.0000 | 0.8257 |

K=7은 K=6 대비 오탐 여유를 5배 벌면서 검출은 K=8부터 시작되는 급락(83%→66%→46%) 전에 머문다.

**이 표는 정직하게 낙관적이다.** 이항 분포는 프레임 오류가 독립이라고 가정하지만 실제로는 아니다. 한
사람이 표정을 유지하는데 모델이 그걸 계속 오독하면 오류는 상관된다. **K-of-N은 산발적 오탐은 잘 막고
체계적 오독은 거의 못 막는다.** 오탐 열은 상한이 아니라 하한으로 읽어야 한다.

`p >= 0.5` 조건은 τ(0.98)로 대체하지 **말 것** — 둘은 다른 일을 한다. τ는 프레임 채택 여부,
K는 이벤트 발생 여부다. 권장 조합: **τ를 통과한 프레임만 K 카운트에 넣는다.**

---

## 6. 이미 맞는 것 — 확인만

| 항목 | 값 | 위치 |
|---|---|---|
| MediaPipe | `model_selection=0`, `min_detection_confidence=0.5` | 학습·서버 동일 |
| 채널 순서 | MediaPipe에 RGB 투입 | `detector.py`가 `bgr[:, :, ::-1]` |
| 그레이스케일 | 배열과 변환 코드가 짝이 맞음 | §3.4 |
| FE 프레임 | **짧은 변 224px, 종횡비 보존** | [`page.tsx` §25,56](../../frontend/app/interview/page.tsx) |
| FE 전송 | 3fps(333ms), JPEG quality 0.7 | [`page.tsx` §26,144](../../frontend/app/interview/page.tsx) |

FE 리사이즈는 **이 프로젝트에서 가장 큰 단일 레버**였다 — 정사각형으로 늘리면 정확도가 0.847 → 0.360으로
무너진다([`04` §3.4](./04-service-calibration.md)). 현재 구현이 옳으므로, 이 코드를 건드리는 사람이
있다면 그 이유를 반드시 알아야 한다. `FRAME_SHORT_PX`에 주석으로 근거를 남겨 둘 것을 권한다.

FE의 JPEG 0.7은 스펙 §7.1 Q3의 답이기도 하다. 학습의 재압축 증강 범위는 q55~92이므로 실제 값을 포함한다.

---

## 7. 성능표 (스펙 §7 항목 5)

**Set A, TEST 2,720프레임, 열화 적용, 서비스 prior 기준** — 배포 조건이다.

| 지표 | 원본 7c | **납품 후보 (ft + bias)** |
|---|---|---|
| accuracy | 0.8193 | **0.8471** |
| macro F1 | 0.7690 | **0.8083** |
| 불안 recall | 0.5187 | **0.6173** |
| 불안 precision | 0.5410 | **0.5802** |
| 중립→불안 오탐 | 0.0765 | **0.0765** |

클래스별(precision / recall): 기쁨 0.976 / 0.917, 당황 0.706 / 0.848, **불안 0.580 / 0.617**,
중립 0.946 / 0.893.

**주장할 수 있는 것은 이게 전부다: 원본 대비 불안 recall +0.099, precision +0.039, 오탐 동률.**

### 7.1 함께 전달해야 하는 단서

이 셋을 빼고 숫자만 전달하면 안 된다.

1. **최종 채택이 아니라 후보다.** 스펙 §6.1의 채택 기준인 Set B(웹캠 녹화)는 여전히 비어 있다. 위 숫자는
   전부 AI-Hub 스틸 이미지 기준이다.
2. **개선폭은 +0.205가 아니다.** 시드 5개에서 불안 recall 개선폭은 **+0.242 ± 0.102(범위 +0.114~+0.393)**
   이다. 개선 자체는 실재하지만(5/5 양수) 단일 수치를 인용하면 안 된다.
3. **ECE는 0.0990이다.** [`04` §1](./04-service-calibration.md)의 "ECE 0.0269, 2.9배 개선"은 **소프트 라벨
   모델의 값이고 이 후보의 값이 아니다.** 이 후보(seed1, hard)는 원본 7c(0.1057)보다 겨우 낫다.

---

## 8. 체크리스트

**추론 서버 (`inference/`)**

- [ ] `emotionnet_v2.pth` + `emotionnet_v2.meta.json`을 `weights/`에 배치 (sha256 대조)
- [ ] `EmotionNet(num_classes=4)`, 라벨 `["기쁨","당황","불안","중립"]`
- [ ] `meta.json`에서 `bias.value`·`tau_biased.value`·`class_order`를 읽어 적용 (하드코딩 금지)
- [ ] `softmax(logprob + bias)` → argmax → τ 비교 순서
- [ ] 얼굴 선택을 검출 점수 최고로 변경
- [ ] 감정 경로에서 정사각형화·패딩 제거 (시선·집중은 유지)
- [ ] 감정 48×48 보간을 `INTER_LINEAR`로 변경
- [ ] 응답에 `accepted` 추가

**백엔드 (`backend/`)**

- [ ] `EMOTION_LABELS` 4클래스로 교체
- [ ] `emotion_negative` 판정을 K=7-of-15로 교체, τ 통과 프레임만 카운트

**AI (남은 일)**

- [ ] Set B 녹화 — 채택 기준. 이것 없이는 위 성능표가 스틸 이미지 기준으로만 유효하다
- [ ] seed1 + 소프트 라벨 재학습 (ECE 0.0990 → 0.0269 기대)
