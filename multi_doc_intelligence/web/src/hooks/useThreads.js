import { useState, useCallback, useEffect } from 'react';
import * as api from '../api/client';

/**
 * Thread CRUD state management for a session.
 */
export function useThreads(sessionId) {
  const [threads, setThreads] = useState([]);
  const [activeThread, setActiveThread] = useState(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const data = await api.fetchThreads(sessionId);
      setThreads(data.threads || []);
      if (data.active_thread && data.threads?.find((t) => t.thread_id === data.active_thread)) {
        setActiveThread(data.active_thread);
      } else if (data.threads?.length) {
        setActiveThread(data.threads[0].thread_id);
      }
    } catch (err) {
      console.error('Failed to fetch threads:', err);
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const create = useCallback(async () => {
    const result = await api.createThread(sessionId);
    setActiveThread(result.thread_id);
    await refresh();
    return result;
  }, [sessionId, refresh]);

  const rename = useCallback(async (threadId, name) => {
    await api.renameThread(sessionId, threadId, name);
    await refresh();
  }, [sessionId, refresh]);

  const remove = useCallback(async (threadId) => {
    await api.deleteThread(sessionId, threadId);
    await refresh();
  }, [sessionId, refresh]);

  return {
    threads,
    activeThread,
    setActiveThread,
    loading,
    create,
    rename,
    remove,
    refresh,
  };
}
