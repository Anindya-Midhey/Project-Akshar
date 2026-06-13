import { useState, useCallback } from 'react';
import ApplyAllButton from './ApplyAllButton';

/**
 * AdjustCorners — ScanTailor-style panel (panel #2)
 * Auto-detects document corners, lets the user drag them on the canvas,
 * then applies perspective warp to crop/straighten the document.
 */
export default function AdjustCorners({
  onDetect,
  onApply,
  onApplyAll,
  loading,
  batchLoading,
  batchProgress,
  hasImage,
  pageCount = 1,
  cornersActive,         // boolean — are corners currently shown on canvas?
  onCornersActiveChange, // (bool) => void — toggle corner overlay visibility
}) {
  const [detected, setDetected] = useState(false);

  const handleDetect = useCallback(async () => {
    const result = await onDetect?.();
    if (result) {
      setDetected(true);
      onCornersActiveChange?.(true);
    }
  }, [onDetect, onCornersActiveChange]);

  const handleApply = useCallback(async () => {
    await onApply?.();
    setDetected(false);
    onCornersActiveChange?.(false);
  }, [onApply, onCornersActiveChange]);

  const handleApplyAll = useCallback(async () => {
    await onApplyAll?.();
    setDetected(false);
    onCornersActiveChange?.(false);
  }, [onApplyAll, onCornersActiveChange]);

  const dis = loading || batchLoading || !hasImage;

  return (
    <>
      <div style={{ textAlign: 'center' }}>
        <div className="field-label" style={{ marginBottom: 10 }}>Corner Detection</div>

        <p style={{
          fontSize: 11, color: 'var(--color-text-3)',
          margin: '0 0 12px', lineHeight: 1.5,
        }}>
          Auto-detect document edges, then drag corners to fine-tune before applying perspective correction.
        </p>

        {/* Detect button */}
        <button
          className="btn btn-sm"
          onClick={handleDetect}
          disabled={dis}
          id="btn-corners-detect"
          style={{
            width: '100%',
            marginBottom: 8,
            background: cornersActive
              ? 'var(--color-surface-3)'
              : 'linear-gradient(135deg, #00e5ff 0%, #7c4dff 100%)',
            color: cornersActive ? 'var(--color-text-2)' : '#fff',
            border: cornersActive ? '1px solid var(--color-cyan)' : 'none',
          }}
        >
          {loading ? (
            <span className="spinner" style={{ width: 14, height: 14, borderWidth: 1.5, marginRight: 6 }} />
          ) : null}
          {cornersActive ? '⟳ Re-detect Corners' : '⊞ Detect Corners'}
        </button>

        {/* Corner info badge */}
        {cornersActive && (
          <div style={{
            fontSize: 11, color: 'var(--color-cyan)',
            marginBottom: 10, fontWeight: 600,
          }}>
            ✓ Corners detected — drag to adjust
          </div>
        )}

        {/* Apply / Hide row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 4 }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => {
              onCornersActiveChange?.(false);
              setDetected(false);
            }}
            disabled={dis || !cornersActive}
            id="btn-corners-hide"
          >✕ Hide</button>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleApply}
            disabled={dis || !cornersActive}
            id="btn-corners-apply"
            style={{
              background: !cornersActive ? 'var(--color-surface-3)' : undefined,
              color: !cornersActive ? 'var(--color-text-3)' : undefined,
            }}
          >
            ✓ Apply
          </button>
        </div>
      </div>

      {/* Apply to All Pages */}
      <ApplyAllButton
        onClick={handleApplyAll}
        pageCount={pageCount}
        disabled={dis || !cornersActive}
        loading={batchLoading}
        progress={batchProgress}
        title={!cornersActive
          ? 'Detect corners first, then apply to all'
          : `Detect & apply corners to all ${pageCount} pages`}
        label={`⊞ Detect & Apply to All ${pageCount} Pages`}
      />
    </>
  );
}
