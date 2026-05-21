/* ═══════════════════════════════════════
   Data Studio — Right Inspector Panel
   Column stats, histogram, frequency bars
   ═══════════════════════════════════════ */
import { useDataStudio } from '../DataStudioStore';
import HistogramCanvas from '../viz/HistogramCanvas';
import BarChartCanvas from '../viz/BarChartCanvas';
import Icon from '../../components/common/Icon';
import type { NumericStats, CategoricalStats } from '../types';

function fmtNum(n: number): string {
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(2) + 'k';
  return n % 1 === 0 ? String(n) : n.toFixed(4);
}

export default function RightPanel() {
  const { state, dispatch } = useDataStudio();
  const { selectedColumn, schema, columnStats, working } = state;

  const cols = schema?.columns.map(c => c.name) || [];

  if (!schema) {
    return (
      <aside className="ds-right">
        <div className="ds-right__header">
          <div className="ds-left__header-row">
            <Icon name="analytics" size={16} style={{ color: 'var(--c-tertiary)' }} />
            <span className="ds-left__header-title">Inspector</span>
          </div>
          <span className="ds-left__header-sub">No data loaded</span>
        </div>
        <div className="ds-empty-state">
          <Icon name="cloud_upload" size={32} />
          <span>Upload data to begin</span>
        </div>
      </aside>
    );
  }

  const renderSelector = () => (
    <div className="ds-field" style={{ marginBottom: 16 }}>
      <span className="ds-field__label">Select Column</span>
      <select
        className="ds-field__select"
        value={selectedColumn || ''}
        onChange={e => dispatch({ type: 'SELECT_COLUMN', col: e.target.value || null })}
      >
        <option value="">— Select —</option>
        {cols.map(c => <option key={c} value={c}>{c}</option>)}
      </select>
    </div>
  );

  if (!selectedColumn) {
    return (
      <aside className="ds-right">
        <div className="ds-right__header">
          <div className="ds-left__header-row">
            <Icon name="analytics" size={16} style={{ color: 'var(--c-tertiary)' }} />
            <span className="ds-left__header-title">Inspector</span>
          </div>
          <span className="ds-left__header-sub">Profiling</span>
        </div>
        <div className="ds-right__body">
          {renderSelector()}
          <div className="ds-empty-state" style={{ marginTop: 40 }}>
            <Icon name="query_stats" size={32} />
            <span>Click a header or use the dropdown to inspect</span>
          </div>
        </div>
      </aside>
    );
  }

  const colSchema = schema.columns.find(c => c.name === selectedColumn);
  const stats = columnStats[selectedColumn];

  return (
    <aside className="ds-right">
      <div className="ds-right__header">
        <div className="ds-left__header-row">
          <Icon name="analytics" size={16} style={{ color: 'var(--c-tertiary)' }} />
          <span className="ds-left__header-title">Inspector</span>
        </div>
        <span className="ds-left__header-sub">Column profiling</span>
      </div>

      <div className="ds-right__body">
        {renderSelector()}

        {/* Column name + type */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
          <span className="ds-right__col-name">{selectedColumn}</span>
          <span className="ds-right__col-type">{colSchema?.type ?? 'unknown'}</span>
        </div>

        {/* Quick stats grid */}
        <div className="ds-stat-grid">
          <div className="ds-stat-card">
            <span className="ds-stat-card__label">Missing</span>
            <span className="ds-stat-card__value">
              {colSchema ? ((colSchema.nullCount / working.length) * 100).toFixed(1) + '%' : '—'}
            </span>
          </div>
          <div className="ds-stat-card">
            <span className="ds-stat-card__label">Unique</span>
            <span className="ds-stat-card__value">
              {colSchema ? fmtNum(colSchema.uniqueCount) : '—'}
            </span>
          </div>
        </div>

        {/* Numeric stats */}
        {stats && stats.type === 'numeric' && (
          <>
            <div className="ds-inspector-viz">
              <span className="ds-inspector-viz__label">Value Distribution</span>
              <HistogramCanvas stats={stats.stats as NumericStats} width={240} height={140} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {[
                ['Mean', fmtNum((stats.stats as NumericStats).mean)],
                ['Median', fmtNum((stats.stats as NumericStats).median)],
                ['Std Dev', fmtNum((stats.stats as NumericStats).stdDev)],
                ['Min', fmtNum((stats.stats as NumericStats).min)],
                ['Max', fmtNum((stats.stats as NumericStats).max)],
                ['Q1', fmtNum((stats.stats as NumericStats).q1)],
                ['Q3', fmtNum((stats.stats as NumericStats).q3)],
                ['Skewness', ((stats.stats as NumericStats).skewness || 0).toFixed(3)],
              ].map(([label, val]) => (
                <div className="ds-stat-row" key={label}>
                  <span className="ds-stat-row__label">{label}</span>
                  <span className="ds-stat-row__value">{val}</span>
                </div>
              ))}
            </div>
          </>
        )}

        {/* Categorical stats */}
        {stats && stats.type === 'categorical' && (
          <>
            <div className="ds-inspector-viz">
              <span className="ds-inspector-viz__label">Top Categories</span>
              <BarChartCanvas
                labels={(stats.stats as CategoricalStats).topN.slice(0, 10).map(t => t.value)}
                values={(stats.stats as CategoricalStats).topN.slice(0, 10).map(t => t.count)}
                width={240}
                height={140}
              />
            </div>

            <div style={{ marginTop: 8 }}>
              <span className="ds-stat-row__label" style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', display: 'block', marginBottom: 6 }}>
                Frequencies
              </span>
              {(stats.stats as CategoricalStats).topN.slice(0, 12).map(({ value, count }) => {
                const maxCount = (stats.stats as CategoricalStats).topN[0]?.count || 1;
                return (
                  <div className="ds-freq-bar" key={value}>
                    <span className="ds-freq-bar__label" title={value}>{value}</span>
                    <div className="ds-freq-bar__track">
                      <div className="ds-freq-bar__fill" style={{ width: `${(count / maxCount) * 100}%` }} />
                    </div>
                    <span className="ds-freq-bar__count">{count}</span>
                  </div>
                );
              })}
            </div>

            <div style={{ marginTop: 8 }}>
              <div className="ds-stat-row">
                <span className="ds-stat-row__label">Mode</span>
                <span className="ds-stat-row__value">{(stats.stats as CategoricalStats).mode}</span>
              </div>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
