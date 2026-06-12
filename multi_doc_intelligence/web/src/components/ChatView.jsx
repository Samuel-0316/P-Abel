import { useState, useRef } from 'react';
import ChatMessage from './ChatMessage';
import { useChat } from '../hooks/useChat';
import { AiIcon, AttachmentIcon, SendIcon, ChatIcon, SearchIcon, DocumentIcon } from './Icons';

export default function ChatView({ sessionId, threadId, onOpenKnowledge, onThreadUpdate }) {
  const { messages, sending, error, send, bottomRef } = useChat(sessionId, threadId, onThreadUpdate);
  const [input, setInput] = useState('');
  const inputRef = useRef(null);

  const hasContent = messages.length > 0 || sending;

  const handleSend = () => {
    if (!input.trim() || sending) return;
    send(input.trim());
    setInput('');
    if (inputRef.current) inputRef.current.style.height = '44px';
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e) => {
    setInput(e.target.value);
    e.target.style.height = '44px';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  // ── No session selected ──────────────────────────────────────────────────
  if (!sessionId) {
    return (
      <div className="chat-container">
        <div className="empty-state" style={{ height: '100vh' }}>
          <div className="empty-state-icon">
            <AiIcon size={26} />
          </div>
          <div className="empty-state-title">Welcome to Multi-Doc Intelligence</div>
          <div className="empty-state-text">
            Create a new Project in the sidebar to start chatting with your documents.
          </div>
        </div>
      </div>
    );
  }

  // ── No thread selected ───────────────────────────────────────────────────
  if (!threadId) {
    return (
      <div className="chat-container">
        <div className="empty-state" style={{ height: '100vh' }}>
          <div className="empty-state-icon">
            <ChatIcon size={24} />
          </div>
          <div className="empty-state-title">No chat selected</div>
          <div className="empty-state-text">
            Click the <strong>+</strong> next to CHATS in the sidebar to start a new conversation.
          </div>
        </div>
      </div>
    );
  }

  // ── Main chat view ────────────────────────────────────────────────────────
  return (
    <div className="chat-container">

      {/* Messages — only rendered when there's content */}
      {hasContent && (
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message-wrapper ${msg.role === 'user' ? 'message-user-wrapper' : ''}`}>
              <ChatMessage message={msg} />
            </div>
          ))}

          {sending && (
            <div className="message-wrapper">
              <div className="message message-assistant">
                <div className="message-avatar">
                  <AiIcon size={16} />
                </div>
                <div className="message-body">
                  <div className="message-content" style={{ paddingTop: 8 }}>
                    <div className="ai-pulse">
                      <div className="ai-pulse-dot" />
                      <div className="ai-pulse-dot" />
                      <div className="ai-pulse-dot" />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} style={{ height: 1 }} />
        </div>
      )}

      {/* ── Floating Input — centered when empty, slides to bottom when active ── */}
      <div className={`chat-input-area ${hasContent ? 'is-bottom' : 'is-centered'}`}>

        {/* Welcome block — only shows when centered */}
        {!hasContent && (
          <div className="chat-welcome">
            <div className="chat-welcome-icon">
              <AiIcon size={28} />
            </div>
            <div className="chat-welcome-title">How can I help you today?</div>
            <div className="chat-welcome-sub">Ask anything about the documents in this project.</div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="text-sm mb-2" style={{ color: 'var(--danger)', textAlign: 'center', marginBottom: 8 }}>
            ⚠️ {error}
          </div>
        )}

        {/* Glass Input Bar */}
        <div className="chat-input-wrapper">
          <button
            className="chat-action-btn"
            title="Manage Project Knowledge"
            onClick={onOpenKnowledge}
            disabled={sending}
          >
            <AttachmentIcon size={17} />
          </button>

          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Ask a question…"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={sending}
          />

          <button
            className={`chat-action-btn ${input.trim() ? 'primary' : ''}`}
            onClick={handleSend}
            disabled={!input.trim() || sending}
          >
            {sending
              ? <div className="spinner" style={{ width: 16, height: 16 }} />
              : <SendIcon size={17} />
            }
          </button>
        </div>

        <div className="chat-input-hint">
          AI can make mistakes. Verify important information with the cited documents.
        </div>
      </div>
    </div>
  );
}
