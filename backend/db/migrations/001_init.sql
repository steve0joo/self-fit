-- SelfFit Phase 1 스키마. Supabase SQL Editor 에서 순서대로 실행.
-- 정본은 이 파일이며 app/models.py 는 이것을 ORM 으로 옮긴 것.

create table if not exists questions (
  id          serial primary key,
  text        text not null,
  category    varchar(50),
  sort_order  int  not null,
  is_active   boolean not null default true
);

create table if not exists sessions (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id) on delete cascade,
  mode        varchar(20) not null default 'live',      -- live | upload
  status      varchar(20) not null default 'created',   -- created | running | finished | failed
  created_at  timestamptz not null default now(),
  started_at  timestamptz,
  finished_at timestamptz
);
create index if not exists sessions_user_created on sessions (user_id, created_at desc);

create table if not exists session_questions (
  id           serial primary key,
  session_id   uuid not null references sessions(id) on delete cascade,
  question_id  int  not null references questions(id),
  order_index  int  not null,
  started_at   timestamptz,
  ended_at     timestamptz,
  unique (session_id, order_index)
);

create table if not exists analysis_logs (
  id              bigserial primary key,
  session_id      uuid not null references sessions(id) on delete cascade,
  ts_ms           int not null,
  question_index  int,
  face_found      boolean,            -- null = 추론 실패
  gaze_yaw        real,
  gaze_pitch      real,
  gaze_state      varchar(10),        -- center | off | unknown
  emotion_probs   jsonb,              -- 모델 출력 그대로
  emotion_top     varchar(10),        -- 중립|불안|당황|기쁨|기타
  attention_probs jsonb,
  attention_top   varchar(10)
);
create index if not exists analysis_logs_session_ts on analysis_logs (session_id, ts_ms);

create table if not exists events (
  id              bigserial primary key,
  session_id      uuid not null references sessions(id) on delete cascade,
  ts_ms           int not null,
  question_index  int,
  type            varchar(30) not null,
  severity        varchar(10) not null,
  message         text not null,
  payload         jsonb
);
create index if not exists events_session_ts on events (session_id, ts_ms);

create table if not exists reports (
  session_id  uuid primary key references sessions(id) on delete cascade,
  summary     jsonb not null,
  created_at  timestamptz not null default now()
);

-- seed: FE 하드코딩 질문과 동일
insert into questions (text, category, sort_order) values
  ('1분간 자기소개를 해주세요.', 'general', 1),
  ('이 직무에 지원한 동기를 말씀해주세요.', 'motivation', 2),
  ('본인의 강점과 약점은 무엇인가요?', 'general', 3),
  ('협업 중 갈등을 해결했던 경험이 있나요?', 'experience', 4),
  ('마지막으로 하고 싶은 말씀이 있다면 해주세요.', 'closing', 5)
on conflict do nothing;
