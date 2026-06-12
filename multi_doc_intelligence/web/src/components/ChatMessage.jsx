import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import TimingBadge from './TimingBadge';
import CitationPanel from './CitationPanel';
import { AiIcon } from './Icons';

/**
 * Single chat message bubble — user or assistant.
 */
export default function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const meta = message.meta || null;

  return (
    <div className={`message ${isUser ? 'message-user' : 'message-assistant'}`}>
      {!isUser && (
        <div className="message-avatar">
          <AiIcon size={16} />
        </div>
      )}
      <div className="message-body">
        <div className="message-content markdown-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        </div>
        {!isUser && meta && (
          <div style={{ marginTop: 8 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <TimingBadge meta={meta} />
              {meta.error && (
                <div className="text-xs" style={{ color: 'var(--accent-danger)' }}>
                  ⚠️ {meta.error}
                </div>
              )}
            </div>
            <CitationPanel meta={meta} />
          </div>
        )}
      </div>
    </div>
  );
}
