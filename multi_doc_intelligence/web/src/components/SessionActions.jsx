import { useState } from 'react';

/**
 * Session action controls — rename, export, delete.
 */
export default function SessionActions({ sessionId, sessionName, onRename, onExport, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(sessionName || '');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [exporting, setExporting] = useState(false);

  if (!sessionId) return null;

  const handleRename = () => {
    if (name.trim()) {
      onRename(sessionId, name.trim());
      setEditing(false);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      await onExport(sessionId);
    } finally {
      setExporting(false);
    }
  };

  const handleDelete = () => {
    if (confirmDelete) {
      onDelete(sessionId);
      setConfirmDelete(false);
    } else {
      setConfirmDelete(true);
      setTimeout(() => setConfirmDelete(false), 3000);
    }
  };

  return (
    <div className="sidebar-section">
      <div className="sidebar-section-title">Session</div>

      {editing ? (
        <div className="inline-edit">
          <input
            className="input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleRename()}
            autoFocus
          />
          <button className="btn btn-sm btn-primary" onClick={handleRename}>✓</button>
          <button className="btn btn-sm" onClick={() => setEditing(false)}>✕</button>
        </div>
      ) : (
        <button className="btn btn-sm btn-full" onClick={() => { setName(sessionName || ''); setEditing(true); }}>
          ✏️ Rename session
        </button>
      )}

      <button className="btn btn-sm btn-full" onClick={handleExport} disabled={exporting}>
        {exporting ? (
          <>
            <div className="spinner" style={{ width: 12, height: 12 }} /> Exporting…
          </>
        ) : (
          '📦 Export session'
        )}
      </button>

      <button
        className={`btn btn-sm btn-full ${confirmDelete ? 'btn-danger' : ''}`}
        onClick={handleDelete}
      >
        {confirmDelete ? '⚠️ Click again to confirm' : '🗑️ Delete session'}
      </button>
    </div>
  );
}
