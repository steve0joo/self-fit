# 05. 모델 성능 비교 리포트

- 작성일: 2026-09-07
- 브랜치: `feature/cv-preprocessing-model`
- 상태: **현재 후보(`deliverable/emotionnet_v2.pth` = seed1 + bias) 채택 근거 정리. 스펙 §6.1의 최종 채택 기준(Set B)은 여전히 미측정 — 이 문서는 "지금 가진 것 중 최선의 후보를 왜 골랐는가"에 대한 답이지 "최종 채택 결정"이 아니다.**
- 선행 문서: [`03-finetuning.md`](./03-finetuning.md), [`04-service-calibration.md`](./04-service-calibration.md), 스펙 [`emotion-finetune-spec.md`](./emotion-finetune-spec.md)
- 데이터 출처: `ai/artifacts/step4_comparison.json` (6모델 × 3경로 × 2prior 전체 격자), `ai/artifacts/deliverable/meta.json` (배포 계약)
- 그림 생성: [`ai/scripts/report_comparison.py`](../scripts/report_comparison.py) → `ai/artifacts/report_comparison/*.png`

이 문서는 새로 학습하거나 재평가하지 않는다. 03·04 문서가 이미 만든 수치를 후보 비교와 채택 근거 중심으로 다시 정리하고, 그 근거가 되는 그림 4장을 추가한 것이다.

---

## 1. 결론 먼저

7개 모델(원본 7클래스 + 파인튜닝 6종: hard-seed0/soft/seed1-4)을 같은 test split·같은 평가 경로에서 비교한 결과:

- **원본 7클래스 대비 파인튜닝은 anxious recall을 일관되게 올린다** (6종 전부 원본 0.519보다 높음). 다만 개선폭은 시드에 따라 +0.114 ~ +0.393으로 넓게 흩어져 있어, 특정 시드 하나(예: 최초 보고된 +0.205)를 대표값으로 인용하면 안 된다.
- **서비스 조건(neutral-dominant prior)에서 재평가하고 클래스별 바이어스까지 튜닝한 뒤 비교하면, `seed1`이 accuracy·anxious precision·macro F1 세 지표 모두에서 6개 후보 중 1위다** (§3 그림 1, §4).
- **현재 배포 후보는 `seed1 + bias([0.0, 1.4, 0.0, 1.9]) + tau(0.98)`다** (`ai/artifacts/deliverable/`). 재학습 비용 없이 체크포인트만 seed0(step3)에서 seed1로 교체한 운영상의 판단이며, **스펙 §6.1이 요구하는 최종 채택 기준(Set B, 팀 웹캠 실측)은 아직 없다.**
- 이 채택은 뒤집힐 수 있다: seed1이 6개 중 최댓값이라는 사실 자체가 "체계적으로 더 나은 초기화"라는 증거가 되지는 않는다 (§5). Set B가 나오면 `checkpoints_step3/`(원래 후보)와 함께 반드시 재검증해야 한다.

---

## 2. 비교 대상

| 이름 | 정체 | 위치 |
|---|---|---|
| `original7` | AI-Hub 원본 7클래스 모델, 4개 열만 추출 | 베이스라인 (파인튜닝 없음) |
| `step3` (hard, seed0) | 최초 파인튜닝 결과. 다수결 hard label, seed 0 | `checkpoints_step3/`, `deliverable_step3/` |
| `soft` | 3인 어노테이터 투표 분포를 그대로 타깃으로 사용 (§4c) | `checkpoints_soft/`, `deliverable_soft/` |
| `seed1`–`seed4` | hard label, seed만 0→1~4로 변경 (선택 지표의 분산 측정용) | `checkpoints_seed{1..4}/` |

전부 같은 전처리 산출물(`X.npy`/`y.npy`/`split.npy`), 같은 test split(2,720장, 28명, train과 인물 분리), 같은 고정 degradation(100px, JPEG 75)에서 평가했다. 라벨 노이즈 상한선(anxious 폴더 순도 37.8%, [[cv-emotion-data-verification]])은 어느 모델도 넘을 수 없는 바닥이라는 점을 모든 비교의 배경으로 깔고 읽을 것.

---

## 3. 비교 1 — 6개 후보 × 4개 핵심 지표 (서비스 조건)

![candidate comparison](../artifacts/report_comparison/candidate_comparison.png)

Set A(원본 해상도 크롭 후 degradation), 서비스 prior(neutral .60/anxious .15/embarrassed .15/happy .10)로 재가중, 각 모델마다 **자기 자신의 val로 다시 튜닝한 바이어스**(neutral→anxious ≤ 0.10 제약 하 anxious recall 최대)를 적용한 결과다 — 바이어스를 하나 정해 모든 모델에 그대로 옮기면 비교 자체가 무효이므로([[cv-emotion-pipeline-plan]]과 같은 원칙), `compare.py`가 후보마다 독립적으로 튜닝했다.

- **accuracy·macro F1·anxious precision 세 지표 모두 `seed1`이 1위**(0.847 / 0.808 / 0.580). `soft`가 근소한 2위(0.846 / 0.808 / 0.568)로 거의 붙어 있다.
- **anxious recall만 보면 `seed3`이 가장 높다**(0.764). 하지만 §5에서 보듯 seed3은 2번째 에폭(학습 초반)에서 뽑힌 결과이고 embarrassed recall이 0.78→0.56으로 무너지는 대가를 치른다 — recall 하나만으로 순위를 매기면 안 되는 이유가 여기 있다.
- **`seed2`는 전 지표에서 최하위다**(accuracy 0.746). 같은 레시피, 같은 데이터로도 시드 하나 차이로 이 정도 편차가 난다는 것 자체가 §5의 "선택 노이즈" 문제를 뒷받침한다.

## 4. 비교 2 — "+0.205"는 대표값이 아니다

![seed variance](../artifacts/report_comparison/seed_variance.png)

원본 7클래스 대비 anxious recall 개선폭을 시드 5개에 대해 그린 것이다(Set A, 균등 prior, 바이어스 없음). 최초 보고(step3=seed0)의 +0.205는 분포의 중간값에 가깝고, 5개 평균은 **+0.242 ± 0.102**, 범위는 **+0.114 ~ +0.393**이다. 5개 전부 양수이므로 "파인튜닝이 anxious recall을 올린다"는 방향성 자체는 견고하지만, 개선폭의 특정 숫자를 성능표 헤드라인으로 단독 인용하면 안 된다 — [`03-finetuning.md`](./03-finetuning.md) §1, [`04-service-calibration.md`](./04-service-calibration.md) §5와 같은 결론이다.

## 5. 비교 3 — 라벨 노이즈를 학습 신호로 쓰면(soft) 보정이 개선된다

![calibration](../artifacts/report_comparison/calibration_ece.png)

같은 시드·같은 split·같은 증강에서 hard label 대신 3인 애노테이터 투표 분포를 그대로 타깃으로 쓰면(§4c), 지표 자체(accuracy/recall/precision)는 0.01~0.03 수준의 작은 변화에 그치지만 **ECE(Expected Calibration Error)는 0.0768 → 0.0269로 2.9배 개선**된다. hard-label 파인튜닝 모델(0.0768)도 원본 7클래스(0.1057)보다는 이미 더 잘 보정돼 있지만, soft label은 거기서 한 번 더 개선한다. 즉 soft label의 이득은 accuracy 표에는 거의 안 잡히고 **확률값의 신뢰도**에 있다 — τ 게이트가 이 확률에 걸리므로, soft 쪽이 "0.5 넘으면 진짜 그럴 확률이 0.5에 가깝다"는 의미를 더 잘 지킨다.

**현재 배포 후보(seed1)는 soft가 아니라 hard label이다** — §3에서 seed1이 soft를 근소하게 이겼기 때문. soft label 모델(`deliverable_soft/`)은 보정 우선순위가 더 중요해지면(예: τ 게이트 오작동이 잦아지면) 재검토할 대안으로 남겨둔다.

## 6. 비교 4 — 채택된 후보의 혼동 행렬, 원본과 나란히

![confusion before after](../artifacts/report_comparison/confusion_before_after.png)

왼쪽은 원본 7클래스(4개 열만 추출), 오른쪽은 **현재 배포 후보**(seed1 + bias `[0.0, 1.4, 0.0, 1.9]`, Set A·균등 prior). 바이어스까지 적용한 뒤에는:

- **anxious recall이 52% → 62%로 회복되는데, 그 회복분은 주로 neutral(24% → 14%)과 happy(5% → 0%)로 새던 프레임을 되찾아온 것이다.** anxious→embarrassed 방향은 오히려 20% → 23%로 소폭 늘었다 — embarrassed/anxious 경계는 라벨 노이즈 상한선(폴더 순도 37.8%)이 걸린 지점이라 완전히 없앨 수 있는 종류가 아니다([[cv-emotion-data-verification]]).
- **neutral recall은 88% → 89%로 거의 그대로 보존된다.** neutral→anxious로 새는 프레임 수 자체가 52건으로 원본과 정확히 같다 — bias가 정확히 이 손실을 되돌리도록 튜닝된 값이라는 뜻이다(bias 없는 원시 seed1은 neutral recall이 0.809로 원본보다 낮았다).
- **대가는 happy 쪽에 있다.** happy recall이 95% → 92%로 소폭 내려가고, 그중 일부(2% → 5%)가 anxious로 잘못 불린다. embarrassed recall은 오히려 78% → 85%로 개선됐다(embarrassed→anxious·embarrassed→neutral 둘 다 줄었다).

---

## 7. 최종 후보로 seed1 + bias를 고른 이유

**"최종 채택"이 아니라 "현재 최선의 후보로 교체"임을 다시 강조한다.** 스펙 §6.1은 Set B(팀 웹캠 실측)를 채택 기준으로 못박았고, 그건 아직 없다(§9). 아래는 Set A/A′ 조건에서 이 체크포인트가 다른 5개 대안보다 나은 근거다.

1. **6개 후보 중 accuracy·anxious precision·macro F1 세 지표 동시 1위** (§3). 어느 한 지표만 좋은 게 아니라 세 지표가 같이 좋다 — seed3처럼 recall 하나만 튀고 나머지가 무너지는 패턴과 다르다.
2. **Set A′(FE→BE 실제 경로, 종횡비 보존)에서도 순위가 유지된다** — Set A에서만 좋은 게 아니라 검출기 열화까지 포함한 조건에서도 같은 우위를 보인다 ([`04-service-calibration.md`](./04-service-calibration.md) §5.1 표).
3. **재학습 비용이 0이다.** 이미 시드 분산을 재려고 5개를 다 학습해 둔 상태였고, seed1은 그중 하나를 canonical 경로로 복사하고 바이어스·τ만 재튜닝하면 됐다. 새로 뭔가를 학습해서 얻은 이득이 아니라 이미 가진 체크포인트 중 재평가로 찾은 이득이다.
4. **트레이드오프 방향이 스펙 §6 우선순위와 일치한다.** anxious recall 1순위가 원본 대비 오르고(0.519→0.617), neutral recall도 거의 보존되며(2순위), 그 비용이 embarrassed↔anxious라는 이미 알려진 라벨 노이즈 경계로 흡수된다 — 새로운 실패 모드가 아니라 기존에 알려진 한계가 조금 커진 것이다.

**다만 이 결론에는 구조적 한계가 있다.**

- **6개 중 최댓값은 그 자체로 최댓값처럼 보이는 게 당연하다.** seed1이 "체계적으로 더 나은 초기화"인지 "이번엔 이 방향으로 운이 좋았던 6개 중 하나"인지는 이 데이터만으로 못 가른다([`04-service-calibration.md`](./04-service-calibration.md) §5.1).
- **전부 AI-Hub 스마트폰 셀피 분포 안에서의 비교다.** 노트북 웹캠, 사무실 조명, 학습 분포 밖 얼굴에 대해서는 어느 모델도 검증되지 않았다 — 이게 바로 Set B가 필요한 이유다.
- **서비스 prior(neutral .60/anxious .15/embarrassed .15/happy .10)는 측정값이 아니라 가정이다.** 바이어스 튜닝 전체가 이 가정에 의존한다.

---

## 8. 배포 계약 (현재 `deliverable/meta.json` 기준)

| 항목 | 값 |
|---|---|
| 가중치 | `deliverable/emotionnet_v2.pth` (seed1, `{'model': state_dict}`) |
| 클래스 순서 | `[happy, embarrassed, anxious, neutral]` |
| 모델 출력 | LogSoftmax (softmax/exp 적용 후 사용) |
| 바이어스 | `[0.0, 1.4, 0.0, 1.9]`, **argmax·τ 비교 이전에 로그확률에 더할 것** |
| τ (바이어스 적용 후) | **0.98**, val coverage 0.643 / test coverage 0.599 |
| K-of-N (제안, 백엔드 미반영) | K=7, N=15 (3fps) — seed1 바이어스 기준 오탐 하한 0.0001 / 검출 92.7% |

바이어스는 이 체크포인트 전용이다 — 다른 시드나 soft 모델에 그대로 옮기면 안 된다(§3의 표에서 후보마다 튜닝된 바이어스가 전부 다른 것을 볼 것).

---

## 9. 남은 일

- [ ] **Set B 녹화·평가** — 스펙 §6.1의 실제 채택 기준. `scripts/setb.py` 준비됨, 녹화 자체가 아직 불가 확인(2026-09-07). **Set B 실행 시 `checkpoints_step3/`(원래 후보)도 반드시 함께 평가할 것** — Set A/A′에서 seed1이 앞섰다고 Set B에서도 앞선다는 보장은 없다.
- [ ] 서비스 prior(§8 가정)를 Set B 실측으로 교체
- [ ] 모델 선택 기준을 "val anxious recall 최대"에서 "제약(neutral→anxious ≤ 상한) 하의 anxious recall 최대"로 바꾼 뒤 시드 분산 재측정 — seed3 같은 실패 모드를 선택 단계에서 걸러낼 수 있는지 확인
