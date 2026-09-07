import { getSupabase } from './supabase';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export async function apiFetch(path: string, init?: RequestInit) {
  const { data } = await getSupabase().auth.getSession();
  const token = data.session?.access_token;
  return fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      ...init?.headers,
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });
}

/** apiFetch와 동일하게 토큰을 붙이되 Content-Type은 지정하지 않는다.
 *  FormData 전송 시 브라우저가 boundary 포함 Content-Type을 직접 설정해야 하기 때문. */
export async function apiFetchRaw(path: string, init?: RequestInit) {
  const { data } = await getSupabase().auth.getSession();
  const token = data.session?.access_token;
  return fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      ...init?.headers,
      Authorization: `Bearer ${token}`,
    },
  });
}

export function getWsUrl(sessionId: string, token: string) {
  return `${API_URL.replace('http', 'ws')}/ws/sessions/${sessionId}?token=${token}`;
}