import { useState, useRef, useEffect } from 'react';
import ThemeToggle from './ThemeToggle';
import { FolderIcon, ChatIcon, EditIcon, TrashIcon, CheckIcon, XIcon, WarningIcon, DotsIcon, SidebarCloseIcon } from './Icons';

/**
 * Sidebar — Projects (Sessions) with nested Chats (Threads).
 * Collapsible: slides off-screen when collapsed, hamburger in header re-opens.
 */
export default function Sidebar({
  theme, onThemeToggle,
  sessions, activeSession, onSelectSession, onCreateSession,
  onRenameSession, onDeleteSession,
  threads, activeThread, onSelectThread,
  onCreateThread, onRenameThread, onDeleteThread,
  collapsed, onToggleCollapse,
}) {
  const [menuOpen, setMenuOpen] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [showNewProject, setShowNewProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const menuRef = useRef(null);
  const newProjectRef = useRef(null);

  // Close menus on outside click
  useEffect(() => {
    const handler = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(null);
        setConfirmDelete(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Auto-focus new project input
  useEffect(() => {
    if (showNewProject && newProjectRef.current) {
      newProjectRef.current.focus();
    }
  }, [showNewProject]);

  const handleNewProject = () => {
    setShowNewProject(true);
    setNewProjectName('');
  };

  const submitNewProject = async () => {
    const name = newProjectName.trim();
    if (!name) return;
    setShowNewProject(false);
    await onCreateSession(name);
  };

  const startRename = (type, id, currentName) => {
    setMenuOpen(null);
    setEditingId(`${type}-${id}`);
    setEditName(currentName);
  };

  const submitRename = (type, id) => {
    const name = editName.trim();
    if (!name) return;
    if (type === 'project') onRenameSession(id, name);
    else onRenameThread(id, name);
    setEditingId(null);
  };

  const handleDelete = (type, id) => {
    if (type === 'project') onDeleteSession(id);
    else onDeleteThread(id);
    setConfirmDelete(null);
    setMenuOpen(null);
  };

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
      {/* Header */}
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <div className="sidebar-logo-icon">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
          </div>
          <div className="sidebar-logo-text">Multi-Doc</div>
        </div>
        <div className="sidebar-header-controls">
          <ThemeToggle theme={theme} onToggle={onThemeToggle} />
          <button
            className="btn-icon sidebar-collapse-btn"
            onClick={onToggleCollapse}
            title={collapsed ? 'Open sidebar' : 'Close sidebar'}
          >
            <SidebarCloseIcon size={18} />
          </button>
        </div>
      </div>

      {/* New Project Button / Input */}
      <div className="sidebar-section" style={{ marginTop: 4 }}>
        {showNewProject ? (
          <div className="new-project-input">
            <input
              ref={newProjectRef}
              className="input"
              placeholder="Project name..."
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitNewProject();
                if (e.key === 'Escape') setShowNewProject(false);
              }}
              onBlur={() => {
                if (!newProjectName.trim()) setShowNewProject(false);
              }}
            />
            <div style={{ display: 'flex', gap: 4, marginTop: 6 }}>
              <button className="btn btn-sm btn-primary" style={{ flex: 1 }} onClick={submitNewProject} disabled={!newProjectName.trim()}>
                Create
              </button>
              <button className="btn btn-sm" onClick={() => setShowNewProject(false)}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button className="btn btn-sm btn-full" onClick={handleNewProject}>
            <span style={{ fontSize: 16, marginRight: 4 }}>+</span> New Project
          </button>
        )}
      </div>

      {/* Projects & Chats */}
      <div className="sidebar-section" style={{ flex: 1, overflowY: 'auto', marginTop: 8 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {sessions.map((s) => {
            const isActive = activeSession === s.session_id;
            const isEditing = editingId === `project-${s.session_id}`;
            const isMenuOpen = menuOpen === `project-${s.session_id}`;

            return (
              <div key={s.session_id} style={{ display: 'flex', flexDirection: 'column' }}>
                {/* Project Item */}
                {isEditing ? (
                  <div className="inline-edit" style={{ padding: '4px 8px' }}>
                    <input
                      className="input"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') submitRename('project', s.session_id);
                        if (e.key === 'Escape') setEditingId(null);
                      }}
                      autoFocus
                    />
                    <button className="btn btn-sm btn-primary" onClick={() => submitRename('project', s.session_id)}>
                      <CheckIcon />
                    </button>
                    <button className="btn btn-sm" onClick={() => setEditingId(null)}>
                      <XIcon />
                    </button>
                  </div>
                ) : (
                  <div
                    className={`list-item ${isActive ? 'active' : ''}`}
                    onClick={() => onSelectSession(s.session_id)}
                  >
                    <FolderIcon size={16} style={{ opacity: 0.8, flexShrink: 0 }} />
                    <span className="list-item-name">{s.session_name}</span>
                    <span className="list-item-meta">{s.document_count}</span>

                    {/* Three-dot menu */}
                    <div className="ctx-menu-anchor" ref={isMenuOpen ? menuRef : null}>
                      <button
                        className="ctx-trigger"
                        onClick={(e) => {
                          e.stopPropagation();
                          setMenuOpen(isMenuOpen ? null : `project-${s.session_id}`);
                          setConfirmDelete(null);
                        }}
                      >
                        <DotsIcon size={14} />
                      </button>
                      {isMenuOpen && (
                        <div className="ctx-menu">
                          <button className="ctx-item" onClick={(e) => { e.stopPropagation(); startRename('project', s.session_id, s.session_name); }}>
                            <EditIcon /> Rename
                          </button>
                          <button
                            className="ctx-item ctx-item-danger"
                            onClick={(e) => { e.stopPropagation(); handleDelete('project', s.session_id); }}
                          >
                            <TrashIcon /> Delete
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Nested Chats */}
                {isActive && !collapsed && (
                  <div className="sidebar-chats-nested" style={{ paddingLeft: 20, marginTop: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '6px 8px 2px' }}>
                      <span style={{ fontSize: 10, color: 'var(--text-tertiary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Chats</span>
                      <button
                        className="new-chat-btn"
                        onClick={(e) => { e.stopPropagation(); onCreateThread(); }}
                        title="New chat"
                      >
                        +
                      </button>
                    </div>
                    {threads.map((t) => {
                      const isChatActive = activeThread === t.thread_id;
                      const isChatEditing = editingId === `chat-${t.thread_id}`;
                      const isChatMenuOpen = menuOpen === `chat-${t.thread_id}`;

                      return isChatEditing ? (
                        <div key={t.thread_id} className="inline-edit" style={{ padding: '2px 6px' }}>
                          <input
                            className="input"
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') submitRename('chat', t.thread_id);
                              if (e.key === 'Escape') setEditingId(null);
                            }}
                            autoFocus
                            style={{ fontSize: 12 }}
                          />
                          <button className="btn btn-sm btn-primary" onClick={() => submitRename('chat', t.thread_id)}>
                            <CheckIcon />
                          </button>
                          <button className="btn btn-sm" onClick={() => setEditingId(null)}>
                            <XIcon />
                          </button>
                        </div>
                      ) : (
                        <div
                          key={t.thread_id}
                          className={`list-item ${isChatActive ? 'active' : ''}`}
                          onClick={(e) => { e.stopPropagation(); onSelectThread(t.thread_id); }}
                          style={{ padding: '5px 8px' }}
                        >
                          <ChatIcon size={14} style={{ opacity: 0.6, flexShrink: 0 }} />
                          <span className="list-item-name" style={{ fontSize: 12 }}>{t.thread_name}</span>

                          {/* Chat three-dot menu */}
                          <div className="ctx-menu-anchor" ref={isChatMenuOpen ? menuRef : null}>
                            <button
                              className="ctx-trigger"
                              onClick={(e) => {
                                e.stopPropagation();
                                setMenuOpen(isChatMenuOpen ? null : `chat-${t.thread_id}`);
                                setConfirmDelete(null);
                              }}
                              style={{ fontSize: 13 }}
                            >
                              <DotsIcon size={13} />
                            </button>
                            {isChatMenuOpen && (
                              <div className="ctx-menu">
                                <button className="ctx-item" onClick={(e) => { e.stopPropagation(); startRename('chat', t.thread_id, t.thread_name); }}>
                                  <EditIcon /> Rename
                                </button>
                                <button
                                  className="ctx-item ctx-item-danger"
                                  onClick={(e) => { e.stopPropagation(); handleDelete('chat', t.thread_id); }}
                                >
                                  <TrashIcon /> Delete
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      );
                    })}
                    {threads.length === 0 && (
                      <div className="text-xs text-muted" style={{ padding: '4px 8px' }}>No chats yet</div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
          {sessions.length === 0 && (
            <div className="text-xs text-muted" style={{ padding: '16px 8px', textAlign: 'center' }}>
              No projects yet
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
