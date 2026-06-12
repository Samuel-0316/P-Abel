import { useState } from 'react';

/**
 * Thread action controls — new thread, rename, delete.
 */
export default function ThreadActions({ threads, activeThread, onCreate, onRename, onDelete }) {
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');

  const handleStartRename = (tid, currentName) => {
    setEditingId(tid);
    setEditName(currentName);
  };

  const handleRename = () => {
    if (editName.trim() && editingId) {
      onRename(editingId, editName.trim());
      setEditingId(null);
    }
  };

  const activeObj = threads.find((t) => t.thread_id === activeThread);

  return (
    <div className="sidebar-section">
      <div className="sidebar-section-title">Active Thread</div>

      {activeObj && (
        <>
          {editingId === activeThread ? (
            <div className="inline-edit">
              <input
                className="input"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleRename()}
                autoFocus
              />
              <button className="btn btn-sm btn-primary" onClick={handleRename}>✓</button>
              <button className="btn btn-sm" onClick={() => setEditingId(null)}>✕</button>
            </div>
          ) : (
            <button
              className="btn btn-sm btn-full"
              onClick={() => handleStartRename(activeThread, activeObj.thread_name)}
            >
              ✏️ Rename thread
            </button>
          )}
        </>
      )}

      <div style={{ display: 'flex', gap: 6 }}>
        <button className="btn btn-sm btn-primary" style={{ flex: 1 }} onClick={onCreate}>
          + New thread
        </button>
        {activeThread && (
          <button className="btn btn-sm btn-danger" onClick={() => onDelete(activeThread)}>
            🗑️
          </button>
        )}
      </div>
    </div>
  );
}
