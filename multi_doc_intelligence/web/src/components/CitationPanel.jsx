import { useState } from 'react';

/**
 * Expandable citation panel with faithfulness meter.
 */
export default function CitationPanel({ meta }) {
  const [open, setOpen] = useState(false);

  if (!meta) return null;
  const { citations = [], faithfulness = {} } = meta;
  if (!citations.length && !faithfulness.confidence) return null;

  const confidence = (faithfulness.confidence || 0) * 100;
  const fillColor = confidence >= 70 ? 'var(--accent-success)' :
                    confidence >= 40 ? 'var(--accent-warning)' :
                    'var(--accent-danger)';

  return (
    <div>
      <button className="citation-toggle" onClick={() => setOpen(!open)}>
        <span>{open ? '▾' : '▸'}</span>
        <span>Sources & confidence</span>
        <span style={{ marginLeft: 'auto', fontWeight: 600 }}>
          {confidence.toFixed(0)}%
        </span>
      </button>

      {open && (
        <div className="citation-panel">
          <div className="faithfulness-bar">
            <span className="text-xs text-muted">Faithfulness</span>
            <div className="faithfulness-meter">
              <div
                className="faithfulness-fill"
                style={{ width: `${confidence}%`, background: fillColor }}
              />
            </div>
            <span className="text-xs" style={{ color: fillColor }}>
              {confidence.toFixed(0)}%
            </span>
          </div>

          {faithfulness.reason && (
            <p className="text-xs text-muted mb-2" style={{ fontStyle: 'italic' }}>
              {cleanReason(faithfulness.reason)}
            </p>
          )}

          {citations.map((cite, i) => (
            <div key={i} className="citation-item">
              <div className="citation-source">
                📄 {cite.source || 'Unknown source'}
              </div>
              <div className="citation-location">
                {cite.page && cite.page !== 'N/A' ? `Page ${cite.page}` : ''}
                {cite.sheet ? ` · Sheet: ${cite.sheet}` : ''}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function cleanReason(reason) {
  if (!reason) return 'Faithfulness score generated from retrieved context.';
  let text = reason.trim();
  if (text.toLowerCase().includes('specifically documents') ||
      text.toLowerCase().includes('provided context documents')) {
    return 'The answer is grounded in the retrieved evidence from your indexed source files.';
  }
  text = text.replace(/\(\s*specifically\s+documents?[^)]*\)/gi, '');
  text = text.replace(/\bdocuments?\s+(\d+[\s,]*)+/gi, 'retrieved evidence ');
  return text.trim() || 'Faithfulness score generated from retrieved context.';
}
