/**
 * API client — thin fetch wrapper for all /api/* endpoints.
 */

const BASE = '/api';

async function request(path, options = {}) {
  const url = `${BASE}${path}`;
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  // Handle streaming responses (ZIP export)
  if (res.headers.get('content-type')?.includes('application/zip')) {
    return res;
  }

  return res.json();
}

// ── Sessions ──────────────────────────────────────────
export const fetchSessions = () => request('/sessions');

export const createSession = (name) =>
  request('/sessions', {
    method: 'POST',
    body: JSON.stringify({ name: name || null }),
  });

export const renameSession = (id, name) =>
  request(`/sessions/${id}/rename`, {
    method: 'PUT',
    body: JSON.stringify({ name }),
  });

export const deleteSession = (id) =>
  request(`/sessions/${id}`, { method: 'DELETE' });

export const exportSession = async (id) => {
  const res = await fetch(`${BASE}/sessions/${id}/export`);
  if (!res.ok) throw new Error('Export failed');
  const blob = await res.blob();
  const disposition = res.headers.get('content-disposition') || '';
  const match = disposition.match(/filename=(.+)/);
  const filename = match ? match[1] : `session_${id}.zip`;
  return { blob, filename };
};

// ── Threads ───────────────────────────────────────────
export const fetchThreads = (sessionId) =>
  request(`/sessions/${sessionId}/threads`);

export const createThread = (sessionId) =>
  request(`/sessions/${sessionId}/threads`, { method: 'POST' });

export const renameThread = (sessionId, threadId, name) =>
  request(`/sessions/${sessionId}/threads/${threadId}/rename`, {
    method: 'PUT',
    body: JSON.stringify({ name }),
  });

export const deleteThread = (sessionId, threadId) =>
  request(`/sessions/${sessionId}/threads/${threadId}`, {
    method: 'DELETE',
  });

export const fetchMessages = (sessionId, threadId) =>
  request(`/sessions/${sessionId}/threads/${threadId}/messages`);

// ── Chat ──────────────────────────────────────────────
export const sendMessage = (sessionId, threadId, question) =>
  request('/chat', {
    method: 'POST',
    body: JSON.stringify({
      session_id: sessionId,
      thread_id: threadId,
      question,
    }),
  });

// ── Upload ────────────────────────────────────────────
export const uploadFiles = async (sessionId, files) => {
  const form = new FormData();
  for (const f of files) {
    form.append('files', f);
  }

  const res = await fetch(`${BASE}/upload?session_id=${sessionId}`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || 'Upload failed');
  }

  return res.json();
};
