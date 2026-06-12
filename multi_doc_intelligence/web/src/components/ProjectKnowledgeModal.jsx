import { useState, useRef, useCallback, useEffect } from 'react';
import { uploadFiles } from '../api/client';
import { DocumentIcon, XIcon, TrashIcon } from './Icons';

export default function ProjectKnowledgeModal({ sessionId, onClose, onUploadComplete }) {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFiles((prev) => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  }, []);

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFiles((prev) => [...prev, ...Array.from(e.target.files)]);
    }
  };

  const removeFile = (index) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (!files.length || uploading || !sessionId) return;
    setUploading(true);
    setResults(null);
    try {
      const data = await uploadFiles(sessionId, files);
      setResults(data.results);
      setFiles([]);
      if (onUploadComplete) onUploadComplete();
    } catch (err) {
      setResults([{ filename: 'Upload Error', status: 'error', message: err.message }]);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">Project Knowledge Base</div>
          <button className="btn-icon" onClick={onClose} disabled={uploading}>
            <XIcon />
          </button>
        </div>

        <div className="modal-body">
          {results && (
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>Upload Results</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {results.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      padding: '8px 12px',
                      borderRadius: 6,
                      fontSize: 13,
                      border: `1px solid ${r.status === 'error' ? 'var(--accent-danger)' : 'var(--accent-success)'}`,
                      backgroundColor: r.status === 'error' ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                    }}
                  >
                    <strong>{r.filename}</strong>: {r.message}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div
            className={`upload-zone ${isDragging ? 'drag-over' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              multiple
              onChange={handleFileSelect}
              accept=".pdf,.docx,.txt,.csv,.xlsx"
            />
            <DocumentIcon size={32} style={{ color: 'var(--text-tertiary)', marginBottom: 8 }} />
            <div className="upload-zone-title">Click or drag documents here</div>
            <div className="text-xs text-muted" style={{ marginTop: 4 }}>
              Supports PDF, DOCX, TXT, CSV, XLSX
            </div>
          </div>

          {files.length > 0 && (
            <div className="upload-file-list" style={{ marginTop: 24 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 8, textTransform: 'uppercase' }}>
                Files to Upload ({files.length})
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {files.map((file, i) => (
                  <div key={i} className="list-item" style={{ background: 'var(--bg-primary)' }}>
                    <span className="list-item-name">{file.name}</span>
                    <button
                      className="btn-icon"
                      style={{ width: 24, height: 24, color: 'var(--accent-danger)' }}
                      onClick={() => removeFile(i)}
                      disabled={uploading}
                    >
                      <TrashIcon />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer" style={{ padding: '16px 24px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
          <button className="btn" onClick={onClose} disabled={uploading}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={handleUpload}
            disabled={files.length === 0 || uploading}
          >
            {uploading ? (
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <div className="ai-pulse" style={{ height: 16 }}>
                  <div className="ai-pulse-dot" style={{ backgroundColor: 'var(--bg-primary)' }}></div>
                  <div className="ai-pulse-dot" style={{ backgroundColor: 'var(--bg-primary)' }}></div>
                  <div className="ai-pulse-dot" style={{ backgroundColor: 'var(--bg-primary)' }}></div>
                </div> 
                <span style={{ marginLeft: 8 }}>Processing...</span>
              </div>
            ) : (
              'Upload to Project'
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
