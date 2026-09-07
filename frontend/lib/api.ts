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