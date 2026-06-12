/**
 * Timing badge — shows ⚡ for cache hits, 🔄 for fresh responses.
 */
export default function TimingBadge({ meta }) {
  if (!meta) return null;

  const { from_cache, elapsed_ms } = meta;

  // Legacy messages from Streamlit may not have timing data
  if (elapsed_ms == null || isNaN(elapsed_ms)) return null;

  if (from_cache) {
    const display = elapsed_ms < 1000
      ? `${Math.round(elapsed_ms)}ms`
      : `${(elapsed_ms / 1000).toFixed(2)}s`;

    return (
      <span className="timing-badge timing-badge-cache">
        ⚡ {display} · from cache
      </span>
    );
  }

  const display = elapsed_ms >= 1000
    ? `${(elapsed_ms / 1000).toFixed(2)}s`
    : `${Math.round(elapsed_ms)}ms`;

  return (
    <span className="timing-badge timing-badge-fresh">
      🔄 {display}
    </span>
  );
}
