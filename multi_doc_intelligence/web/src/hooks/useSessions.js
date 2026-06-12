import { useState, useCallback, useEffect } from 'react';
import * as api from '../api/client';

/**
 * Session CRUD state management.
 */
export function useSessions() {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = await api.fetchSessions();
      setSessions(data);
    } catch (err) {
      console.error('Failed to fetch sessions:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const create = useCallback(async (name) => {
    const result = await api.createSession(name);
    setActiveSession(result.session_id);
    await refresh();
    return result;
  }, [refresh]);

  const rename = useCallback(async (id, name) => {
    await api.renameSession(id, name);
    await refresh();
  }, [refresh]);

  const remove = useCallback(async (id) => {
    await api.deleteSession(id);
    if (activeSession === id) {
      setActiveSession(null);
    }
    await refresh();
  }, [activeSession, refresh]);

  const exportSession = useCallback(async (id) => {
    const { blob, filename } = await api.exportSession(id);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  return {
    sessions,
    activeSession,
    setActiveSession,
    loading,
    create,
    rename,
    remove,
    exportSession,
    refresh,
  };
}
