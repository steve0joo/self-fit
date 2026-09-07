import { readFileSync, writeFileSync, appendFileSync } from "node:fs";

// 사용법: node scripts/ai-review.mjs <diff 파일> <출력 마크다운 파일>
const [, , diffPath, outPath] = process.argv;
const MODEL = process.env.OPENAI_MODEL || "gpt-4o";
const API_KEY = process.env.OPENAI_API_KEY;
const MAX_DIFF_CHARS = 100_000; // 토큰 폭증/한도 초과 방지용 상한

const MARKER = "<!-- ai-pr-review -->";

// 각 축의 정의와 만점. total = 100.
const AXES = [
  { key: "security", label: "🔒 Security", max: 30 },
  { key: "scope", label: "📦 Scope", max: 20 },
  { key: "breaking", label: "💥 Breaking", max: 20 },
  { key: "test", label: "🧪 Test(부재)", max: 15 },
  { key: "migration", label: "🚚 Migration", max: 15 },
];

const RUBRIC = `너는 시니어 코드 리뷰어다. 아래 PR diff를 리뷰하고 5개 축으로 리스크를 채점하라.
점수는 "리스크"를 나타내며 높을수록 위험(사람이 더 꼼꼼히 봐야 함)이다.

[채점 루브릭]
1) security (0~30): auth/권한/시크릿/암호화 관련 파일 변경, 새 외부 입력 경로 추가, CVE 있는 의존성.
   - 0: 보안 영역 안 건드림 / 11~20: auth·권한 로직 또는 새 입력 경로 1개 / 21~30: 인증·암호화·시크릿 핵심 변경 또는 CVE 의존성.
2) scope (0~20): 변경된 파일 수·라인 수·영향 모듈 수. 아래 "변경 통계"를 신뢰해 채점하라.
   - 0~4: ≤5파일 & ≤100줄 / 10~14: ~16~30파일 또는 500~1500줄 / 15~20: >30파일 또는 >1500줄.
3) breaking (0~20): public API 시그니처 변경, DB 스키마 변경, 환경변수 추가/제거, deprecated 마킹.
   - 0: 하위호환 유지(순수 추가) / 8~14: 기존 public API·스키마 변경 / 15~20: 다수의 호환성 파괴.
4) test (0~15): 테스트 "부재"로 인한 리스크. 테스트가 없을수록 높다.
   - 0: 변경에 상응하는 충분한 테스트 동반 / 6~10: 핵심 로직 테스트 거의 없음 / 11~15: 테스트 전무 + 기존 테스트 삭제.
5) migration (0~15): DB 마이그레이션, 데이터 백필, 인프라/설정 변경. 롤백이 어려울수록 높다.
   - 0: 없음 / 7~11: 롤백 가능한 스키마 마이그레이션 / 12~15: 비가역(컬럼 삭제·대량 백필).

각 축은 정수, 해당 축 만점을 넘기지 마라. reasons는 한국어 한 줄로 근거를 적어라.
findings는 구체적 파일/라인 기반의 실질적 지적만 담아라(없으면 빈 배열).`;

function fail(msg) {
  console.error(msg);
  process.exit(1);
}

if (!API_KEY) fail("OPENAI_API_KEY가 설정되지 않았습니다.");
if (!diffPath || !outPath) fail("사용법: node ai-review.mjs <diff파일> <출력파일>");

let diff = readFileSync(diffPath, "utf8");

// 변경 통계는 잘라내기 전 원본 diff에서 계산해 scope 채점 근거로 넘긴다.
const fileCount = (diff.match(/^diff --git /gm) || []).length;
const added = (diff.match(/^\+(?!\+\+)/gm) || []).length;
const removed = (diff.match(/^-(?!--)/gm) || []).length;
const stats = `파일 ${fileCount}개, +${added} / -${removed} 라인`;

if (diff.trim() === "") {
  writeFileSync(
    outPath,
    `${MARKER}\n## 🤖 AI PR Review\n\n리뷰할 코드 변경이 없습니다 (lock 파일 등 제외됨).\n`
  );
  if (process.env.GITHUB_OUTPUT) {
    appendFileSync(process.env.GITHUB_OUTPUT, `score=0\nlabel=risk:low\n`);
  }
  process.exit(0);
}

let truncated = false;
if (diff.length > MAX_DIFF_CHARS) {
  diff = diff.slice(0, MAX_DIFF_CHARS);
  truncated = true;
}

const userContent = `변경 통계: ${stats}\n\n[PR DIFF]\n${diff}`;

const schemaHint = `반드시 아래 JSON 형식으로만 응답하라:
{
  "scores": { "security": 0, "scope": 0, "breaking": 0, "test": 0, "migration": 0 },
  "reasons": { "security": "", "scope": "", "breaking": "", "test": "", "migration": "" },
  "summary": "한국어 한 문단 총평",
  "findings": [ { "severity": "high|medium|low", "location": "file:line", "comment": "지적 내용" } ]
}`;

async function callOpenAI() {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      temperature: 0.2,
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: `${RUBRIC}\n\n${schemaHint}` },
        { role: "user", content: userContent },
      ],
    }),
  });

  if (!res.ok) {
    fail(`OpenAI API 오류 ${res.status}: ${await res.text()}`);
  }
  const data = await res.json();
  return JSON.parse(data.choices[0].message.content);
}

const clamp = (n, max) => Math.max(0, Math.min(max, Math.round(Number(n) || 0)));

// 점수 구간 → 라벨/후속 액션. total 0~100 기준.
const BANDS = [
  { max: 15, name: "Low", label: "risk:low", emoji: "🟢", action: "1명 승인으로 auto-merge 후보" },
  { max: 35, name: "Medium", label: "risk:medium", emoji: "🟡", action: "사람 리뷰어 1명 필수 · auto-merge 비활성" },
  { max: 60, name: "High", label: "risk:high", emoji: "🟠", action: "시니어 1명 + Security/Architecture 리뷰 필요" },
  { max: 100, name: "Critical", label: "risk:critical", emoji: "🔴", action: "즉시 사람 호출 · RFC/ADR 링크 필수 · merge 차단" },
];

function bandFor(total, scores) {
  const band = BANDS.find((b) => total <= b.max) || BANDS[BANDS.length - 1];
  // 단일 축 에스컬레이션: 보안/마이그레이션이 치명적이면 최소 High로 승격.
  const forceHigh = scores.security >= 21 || scores.migration >= 12;
  const highIdx = BANDS.findIndex((b) => b.name === "High");
  if (forceHigh && BANDS.indexOf(band) < highIdx) {
    return { ...BANDS[highIdx], escalated: true };
  }
  return band;
}

const SEV_EMOJI = { high: "🔴", medium: "🟡", low: "🔵" };

function render(result, computed) {
  const { scores, total, band } = computed;

  const rows = AXES.map((a) => {
    const reason = (result.reasons?.[a.key] || "").trim() || "—";
    return `| ${a.label} | ${scores[a.key]} / ${a.max} | ${reason} |`;
  }).join("\n");

  let md = `${MARKER}\n`;
  md += `## 🤖 AI PR Review — Risk Score: ${total} / 100 · ${band.emoji} ${band.name}\n\n`;
  if (band.escalated) {
    md += `> ⚠️ 총점은 낮지만 보안/마이그레이션 축이 높아 **High로 승격**되었습니다.\n\n`;
  }
  md += `**다음 단계:** ${band.action}\n\n`;
  md += `| 축 | 점수 | 근거 |\n|---|---|---|\n${rows}\n\n`;
  if (result.summary) md += `${result.summary.trim()}\n\n`;

  const findings = Array.isArray(result.findings) ? result.findings : [];
  if (findings.length) {
    md += `### 상세 리뷰\n`;
    for (const f of findings) {
      const emoji = SEV_EMOJI[f.severity] || "⚪";
      const loc = f.location ? `\`${f.location}\` — ` : "";
      md += `- ${emoji} ${loc}${(f.comment || "").trim()}\n`;
    }
    md += `\n`;
  } else {
    md += `### 상세 리뷰\n특별히 지적할 사항을 찾지 못했습니다.\n\n`;
  }

  md += `<sub>모델: ${MODEL} · 변경 통계: ${stats}`;
  if (truncated) md += ` · ⚠️ diff가 커서 일부만 리뷰됨(${MAX_DIFF_CHARS.toLocaleString()}자 상한)`;
  md += `</sub>\n`;
  return md;
}

function computeScores(result) {
  const scores = {};
  for (const a of AXES) scores[a.key] = clamp(result.scores?.[a.key], a.max);
  const total = AXES.reduce((s, a) => s + scores[a.key], 0);
  return { scores, total, band: bandFor(total, scores) };
}

const result = await callOpenAI();
const computed = computeScores(result);
writeFileSync(outPath, render(result, computed));

// GitHub Actions 후속 스텝(라벨 부여/게이트)에서 쓰도록 출력값을 넘긴다.
if (process.env.GITHUB_OUTPUT) {
  appendFileSync(
    process.env.GITHUB_OUTPUT,
    `score=${computed.total}\nlabel=${computed.band.label}\n`
  );
}
console.log(`리뷰 생성 완료: ${outPath} (score=${computed.total}, ${computed.band.label})`);
