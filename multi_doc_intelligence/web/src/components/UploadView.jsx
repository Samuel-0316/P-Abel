import { useState, useRef, useCallback } from 'react';
import { uploadFiles } from '../api/client';

/**
 * Upload view — drag-and-drop zone + file list with indexing status.
 */
export default function UploadView({ sessionId, onUploadComplete }) {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const ACCEPTED = ['.pdf', '.docx', '.txt', '.xlsx'];

  const handleFiles = useCallback((fileList) => {
    const valid = Array.from(fileList).filter((f) =>
      ACCEPTED.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      return [...prev, ...valid.filter((f) => !existing.has(f.name))];
    });
    setResults([]);
  }, []);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const removeFile = (name) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const handleUpload = async () => {
    if (!files.length || !sessionId) return;
    setUploading(true);
    setResults([]);

    try {
      const data = await uploadFiles(sessionId, files);
      setResults(data.results || []);
      setFiles([]);
      onUploadComplete?.();
    } catch (err) {
      setResults([{ filename: 'Upload', status: 'error', message: err.message }]);
    } finally {
      setUploading(false);
    }
  };

  if (!sessionId) {
    return (
      <div className="empty-state" style={{ padding: '64px 32px' }}>
        <div className="empty-state-icon">📁</div>
        <div className="empty-state-title">No session selected</div>
        <div className="empty-state-text">
          Create or select a session first to upload documents.
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px 32px', maxWidth: 720 }}>
      <div
        className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <div className="upload-zone-icon">📄</div>
        <div className="upload-zone-title">Drop files here or click to browse</div>
        <div className="upload-zone-subtitle">
          Supports PDF, DOCX, TXT, and XLSX files
        </div>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.txt,.xlsx"
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {files.length > 0 && (
        <>
          <div className="upload-file-list">
            {files.map((f) => (
              <div key={f.name} className="upload-file-item">
                <span className="upload-file-name">📎 {f.name}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className="text-xs text-muted">
                    {(f.size / 1024).toFixed(0)} KB
                  </span>
                  <button
                    className="btn btn-sm btn-danger"
                    onClick={(e) => { e.stopPropagation(); removeFile(f.name); }}
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>

          <button
            className="btn btn-primary btn-full mt-4"
            onClick={handleUpload}
            disabled={uploading}
            style={{ padding: '12px 24px', fontSize: 14 }}
          >
            {uploading ? (
              <>
                <div className="spinner" style={{ borderTopColor: 'white', width: 16, height: 16 }} />
                Indexing files…
              </>
            ) : (
              `Index ${files.length} file${files.length > 1 ? 's' : ''}`
            )}
          </button>
        </>
      )}

      {results.length > 0 && (
        <div className="upload-file-list mt-4">
          {results.map((r, i) => (
            <div key={i} className="upload-file-item">
              <span className="upload-file-name">
                {r.status === 'indexed' ? '✅' : r.status === 'skipped' ? '⏭️' : '❌'}{' '}
                {r.filename}
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className={`upload-file-status ${r.status}`}>
                  {r.status}
                </span>
                {r.message && (
                  <span className="text-xs text-muted">{r.message}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
