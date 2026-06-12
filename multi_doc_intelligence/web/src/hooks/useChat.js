import { useState, useCallback, useEffect, useRef } from 'react';
import * as api from '../api/client';

/**
 * Chat messages + send state for a thread.
 */
export function useChat(sessionId, threadId, onThreadUpdate) {
  const [messages, setMessages] = useState([]);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);

  // Load messages when thread changes
  useEffect(() => {
    if (!sessionId || !threadId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const data = await api.fetchMessages(sessionId, threadId);
        if (!cancelled) setMessages(data.messages || []);
      } catch (err) {
        console.error('Failed to load messages:', err);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId, threadId]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = useCallback(async (question) => {
    if (!question.trim() || sending) return;

    const isFirstMessage = messages.length === 0;

    setError(null);
    const userMsg = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMsg]);
    setSending(true);

    try {
      const result = await api.sendMessage(sessionId, threadId, question);
      const assistantMsg = {
        role: 'assistant',
        content: result.answer,
        meta: {
          citations: result.citations,
          faithfulness: result.faithfulness,
          from_cache: result.from_cache,
          elapsed_ms: result.elapsed_ms,
          error: result.error,
          hyde_query: result.hyde_query,
        },
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // If this was the first message, the backend auto-renamed the thread.
      // Notify the parent to refresh the thread list.
      if (isFirstMessage && onThreadUpdate) {
        onThreadUpdate();
      }
    } catch (err) {
      setError(err.message);
      // Remove the optimistic user message on error
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setSending(false);
    }
  }, [sessionId, threadId, sending, messages.length, onThreadUpdate]);

  return { messages, sending, error, send, bottomRef };
}
