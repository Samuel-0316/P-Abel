import { useState } from 'react';
import { useTheme } from './hooks/useTheme';
import { useSessions } from './hooks/useSessions';
import { useThreads } from './hooks/useThreads';
import Sidebar from './components/Sidebar';
import ChatView from './components/ChatView';
import ProjectKnowledgeModal from './components/ProjectKnowledgeModal';
import { DocumentIcon } from './components/Icons';

export default function App() {
  const { theme, toggle: toggleTheme } = useTheme();
  const {
    sessions, activeSession, setActiveSession,
    create: createSession, rename: renameSession,
    remove: deleteSession, exportSession, refresh: refreshSessions,
  } = useSessions();

  const {
    threads, activeThread, setActiveThread,
    create: createThread, rename: renameThread,
    remove: deleteThread, refresh: refreshThreads,
  } = useThreads(activeSession);

  const [showKnowledgeModal, setShowKnowledgeModal] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const handleCreateSession = async (name) => {
    const result = await createSession(name);
    setActiveSession(result.session_id);
  };

  const handleDeleteSession = async (id) => {
    await deleteSession(id);
    const remaining = sessions.filter((s) => s.session_id !== id);
    setActiveSession(remaining.length > 0 ? remaining[0].session_id : null);
  };

  const activeSessionObj = sessions.find((s) => s.session_id === activeSession);

  return (
    <div className={`app-layout ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <Sidebar
        theme={theme}
        onThemeToggle={toggleTheme}
        sessions={sessions}
        activeSession={activeSession}
        onSelectSession={setActiveSession}
        onCreateSession={handleCreateSession}
        onRenameSession={renameSession}
        onDeleteSession={handleDeleteSession}
        threads={threads}
        activeThread={activeThread}
        onSelectThread={setActiveThread}
        onCreateThread={createThread}
        onRenameThread={renameThread}
        onDeleteThread={deleteThread}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed((v) => !v)}
      />

      <main className="main-content">
        <div className="page-header">
          <div style={{ flex: 1 }} />
          {activeSession && (
            <button
              className="btn btn-primary"
              onClick={() => setShowKnowledgeModal(true)}
              style={{ borderRadius: '20px', padding: '6px 16px' }}
            >
              <DocumentIcon /> Project Knowledge ({activeSessionObj?.document_count || 0})
            </button>
          )}
        </div>

        <ChatView
          sessionId={activeSession}
          threadId={activeThread}
          onOpenKnowledge={() => setShowKnowledgeModal(true)}
          onThreadUpdate={refreshThreads}
        />

        {showKnowledgeModal && (
          <ProjectKnowledgeModal
            sessionId={activeSession}
            onClose={() => setShowKnowledgeModal(false)}
            onUploadComplete={refreshSessions}
          />
        )}
      </main>
    </div>
  );
}
