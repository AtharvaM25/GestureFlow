// Typed client for the GestureFlow API (docs/API.md).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const V1 = `${API_URL}/api/v1`;

export type User = { id: number; email: string; created_at: string };
export type Alternative = { gesture: string; confidence: number };
export type Gesture = { label: string; f1_unseen_session: number | null };
export type Session = {
  id: number;
  text: string;
  sentence: string | null;
  frame_count: number;
  started_at: string;
  ended_at: string | null;
  model_version: string;
};
export type CommittedLetter = {
  id: number;
  gesture: string;
  confidence: number;
  latency_ms: number;
  alternatives: Alternative[];
  created_at: string;
};
export type SessionDetail = Session & { letters: CommittedLetter[] };
export type Health = {
  status: string;
  model_loaded: boolean;
  model_classes: number;
  database: string;
  sentence_provider: "ollama" | "groq" | "none";
};
export type Analytics = {
  days: number;
  sessions: number;
  letters: number;
  frames: number;
  avg_confidence: number | null;
  avg_latency_ms: number | null;
  per_day: { date: string; letters: number }[];
  per_letter: { gesture: string; count: number; avg_confidence: number }[];
};
export type CalibrationStatus = {
  calibrated: boolean;
  stale: boolean;
  samples_per_label: number | null;
  created_at: string | null;
  labels: string[];
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// The login token is kept in localStorage so a page reload keeps you signed in. It expires
// after 24 hours; after that the API answers 401 and you sign in again.
const KEY = "gestureflow.token";

export const tokenStore = {
  get(): string | null {
    try {
      return typeof window === "undefined" ? null : window.localStorage.getItem(KEY);
    } catch {
      return null;
    }
  },
  set(token: string | null) {
    try {
      if (token) window.localStorage.setItem(KEY, token);
      else window.localStorage.removeItem(KEY);
    } catch {
      /* storage unavailable: stay signed in for this page only */
    }
  },
};

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = tokenStore.get();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body) headers.set("Content-Type", "application/json");

  let res: Response;
  try {
    res = await fetch(`${V1}${path}`, { ...init, headers });
  } catch {
    // free hosts put the API to sleep when idle; the first request wakes it
    throw new ApiError(0, "Can't reach the server. If it was asleep it can take up to a minute to start; try again shortly.");
  }
  const body = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) {
    if (res.status === 401) tokenStore.set(null); // expired or invalid: sign in again
    const detail = body?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: { msg: string }) => d.msg).join("; ")
          : `request failed (${res.status})`;
    throw new ApiError(res.status, message);
  }
  return body as T;
}

const post = (body?: unknown): RequestInit => ({ method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health"),
  register: (email: string, password: string) => request<User>("/auth/register", post({ email, password })),
  async login(email: string, password: string) {
    const { access_token } = await request<{ access_token: string }>("/auth/login", post({ email, password }));
    tokenStore.set(access_token);
  },
  logout: () => tokenStore.set(null),
  me: () => request<User>("/auth/me"),
  deleteAccount: (password: string) => request<void>("/auth/me", { method: "DELETE", body: JSON.stringify({ password }) }),
  gestures: () => request<Gesture[]>("/gestures"),
  sessions: () => request<Session[]>("/sessions"),
  session: (id: number) => request<SessionDetail>(`/sessions/${id}`),
  createSession: () => request<Session>("/sessions", post()),
  endSession: (id: number) => request<Session>(`/sessions/${id}/end`, post()),
  sentence: (id: number) => request<{ sentence: string }>(`/sessions/${id}/sentence`, post()),
  analytics: (days: number) => request<Analytics>(`/analytics?days=${days}`),
  calibration: () => request<CalibrationStatus>("/calibration"),
  saveCalibration: (samples: Record<string, number[][][]>) =>
    request<CalibrationStatus>("/calibration", post({ samples })),
  deleteCalibration: () => request<void>("/calibration", { method: "DELETE" }),
};

export const wsUrl = () => `${API_URL.replace(/^http/, "ws")}/api/v1/ws/recognition`;
