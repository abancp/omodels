/* ═══════════════════════════════════════
   Data Studio — Center Content
   Dropzone, data table, viz area, transform/split/schema sub-panels
   ═══════════════════════════════════════ */
import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import Papa from 'papaparse';
import { useDataStudio } from '../DataStudioStore';
import { computeCorrelationMatrix } from '../engine';
import HistogramCanvas from '../viz/HistogramCanvas';
import ScatterCanvas from '../viz/ScatterCanvas';
import CorrelationHeatmap from '../viz/CorrelationHeatmap';
import BoxPlotCanvas from '../viz/BoxPlotCanvas';
import BarChartCanvas from '../viz/BarChartCanvas';
import MissingMapCanvas from '../viz/MissingMapCanvas';
import Icon from '../../components/common/Icon';
import { faker } from '@faker-js/faker';
import type { TransformKind, VizType, NumericStats, CategoricalStats } from '../types';

const MAX_TABLE_ROWS = 200;

/* ─── Transform definitions ─── */
const TRANSFORM_DEFS: { kind: TransformKind; label: string; needsCol: boolean; extra?: string[] }[] = [
  { kind: 'drop-column', label: 'Drop Column', needsCol: true },
  { kind: 'rename-column', label: 'Rename Column', needsCol: true, extra: ['newName'] },
  { kind: 'drop-nulls', label: 'Drop Nulls', needsCol: false },
  { kind: 'fill-nulls', label: 'Fill Nulls', needsCol: true, extra: ['method'] },
  { kind: 'min-max-scale', label: 'Min-Max Scale', needsCol: true },
  { kind: 'z-score', label: 'Z-Score Norm', needsCol: true },
  { kind: 'log-transform', label: 'Log Transform', needsCol: true },
  { kind: 'one-hot-encode', label: 'One-Hot Encode', needsCol: true },
  { kind: 'label-encode', label: 'Label Encode', needsCol: true },
  { kind: 'drop-duplicates', label: 'Drop Duplicates', needsCol: false },
  { kind: 'clip-outliers', label: 'Clip Outliers', needsCol: true },
  { kind: 'bin-numeric', label: 'Bin Numeric', needsCol: true, extra: ['bins'] },
  { kind: 'sort-by', label: 'Sort By', needsCol: true, extra: ['ascending'] },
  { kind: 'filter-rows', label: 'Filter Rows', needsCol: true, extra: ['op', 'value'] },
];

const FILL_METHODS = ['mean', 'median', 'mode', 'zero', 'custom'];
const FILTER_OPS = ['==', '!=', '>', '<', '>=', '<=', 'contains'];

const VIZ_OPTIONS: { id: VizType; label: string; icon: string }[] = [
  { id: 'histogram', label: 'Histogram', icon: 'bar_chart' },
  { id: 'scatter', label: 'Scatter', icon: 'scatter_plot' },
  { id: 'correlation', label: 'Correlation', icon: 'grid_on' },
  { id: 'box-plot', label: 'Box Plot', icon: 'candlestick_chart' },
  { id: 'bar-chart', label: 'Bar Chart', icon: 'bar_chart' },
  { id: 'missing-map', label: 'Missing Map', icon: 'view_module' },
];

/* ─── Dropzone ─── */
function Dropzone() {
  const { importFile } = useDataStudio();
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files[0]) importFile(e.dataTransfer.files[0]);
  }, [importFile]);

  const handleFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) importFile(e.target.files[0]);
  }, [importFile]);

  return (
    <div
      className={`ds-dropzone ${dragActive ? 'ds-dropzone--active' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept=".csv,.json,.jsonl,.tsv" hidden onChange={handleFile} />
      <div className="ds-dropzone__icon">
        <Icon name="upload_file" size={36} />
      </div>
      <span className="ds-dropzone__title">Import Dataset</span>
      <span className="ds-dropzone__sub">Drag & drop CSV, JSON, or JSONL files here</span>
      <button className="ds-dropzone__btn" onClick={e => { e.stopPropagation(); inputRef.current?.click(); }}>
        Browse Files
      </button>
    </div>
  );
}

/* ─── Data Table ─── */
function DataTable({ search }: { search?: any }) {
  const { state, dispatch } = useDataStudio();
  const { working, schema, selectedColumn } = state;
  const [editing, setEditing] = useState<{ row: number; col: string } | null>(null);
  const [editVal, setEditVal] = useState('');
  const [confirmDialog, setConfirmDialog] = useState<{ show: boolean; msg: string; onConfirm: () => void } | null>(null);

  const highlightClass = (val: any) => {
    if (!search?.find) return '';
    return String(val) === search.find ? 'ds-table__td--highlight' : '';
  };

  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; rowIdx?: number; col?: string } | null>(null);

  const handleContextMenu = (e: React.MouseEvent, rowIdx?: number, col?: string) => {
    if (state.activePanel !== 'edit') return;
    e.preventDefault();
    setCtxMenu({ x: e.clientX, y: e.clientY, rowIdx, col });
  };

  useEffect(() => {
    const hide = () => setCtxMenu(null);
    window.addEventListener('click', hide);
    return () => window.removeEventListener('click', hide);
  }, []);

  if (!schema) return null;
  const cols = schema.columns.map(c => c.name);
  const displayRows = working.slice(0, MAX_TABLE_ROWS);

  const commitEdit = (rowIdx: number, col: string, val: string) => {
    const colSchema = schema.columns.find(c => c.name === col);
    if (colSchema?.type === 'numeric' && isNaN(Number(val)) && val !== '') {
      setConfirmDialog({
        show: true,
        msg: `Value "${val}" is not numeric. Column "${col}" is typed as numeric. Confirm forcing this as a string?`,
        onConfirm: () => {
          dispatch({ type: 'EDIT_CELL', rowIdx, column: col, value: val });
          setEditing(null);
          setConfirmDialog(null);
        }
      });
      return;
    }
    dispatch({ type: 'EDIT_CELL', rowIdx, column: col, value: val });
    setEditing(null);
  };

  return (
    <div className="ds-table-wrap">
      <table className={`ds-table ${state.activePanel === 'edit' ? 'ds-table--editable' : ''}`}>
        <thead>
          <tr>
            <th className="ds-table__row-num">#</th>
            {cols.map(col => (
              <th key={col} className={selectedColumn === col ? 'ds-table__th--selected' : ''} onClick={() => dispatch({ type: 'SELECT_COLUMN', col })} onContextMenu={(e) => handleContextMenu(e, undefined, col)}>
                {col}
                <span className="ds-table__type-badge">{schema.columns.find(c => c.name === col)?.type}</span>
              </th>
            ))}
            <th style={{ width: 40 }} />
          </tr>
        </thead>
        <tbody>
          {displayRows.map((row, idx) => (
            <tr key={idx}>
              <td className="ds-table__row-num">{idx + 1}</td>
              {cols.map(col => (
                <td key={col} className={highlightClass(row[col])} onContextMenu={(e) => handleContextMenu(e, idx, col)} onDoubleClick={() => { 
                  if (state.activePanel === 'edit') {
                    setEditing({ row: idx, col }); 
                    setEditVal(String(row[col] ?? '')); 
                  }
                }}>
                  {editing?.row === idx && editing.col === col ? (
                    <input className="ds-table__edit-input" autoFocus value={editVal} onChange={e => setEditVal(e.target.value)} 
                      onBlur={() => commitEdit(idx, col, editVal)} onKeyDown={e => { if (e.key === 'Enter') commitEdit(idx, col, editVal); }} />
                  ) : (
                    String(row[col] ?? '')
                  )}
                </td>
              ))}
              <td className="ds-table__row-actions">
                <button onClick={() => dispatch({ type: 'DUPLICATE_ROW', rowIdx: idx })}><Icon name="content_copy" size={10} /></button>
                <button onClick={() => dispatch({ type: 'DELETE_ROW', rowIdx: idx })}><Icon name="delete" size={10} /></button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {confirmDialog?.show && (
        <div className="ds-dialog-overlay">
          <div className="ds-dialog">
            <h3>Type Evolution Warning</h3>
            <p>{confirmDialog.msg}</p>
            <div className="ds-dialog__actions">
              <button className="ds-btn ds-btn--ghost" onClick={() => setConfirmDialog(null)}>Cancel</button>
              <button className="ds-btn ds-btn--primary" onClick={confirmDialog.onConfirm}>Confirm Change</button>
            </div>
          </div>
        </div>
      )}

      {ctxMenu && (
        <div className="ds-ctx-menu" style={{ top: ctxMenu.y, left: ctxMenu.x }}>
          <div className="ds-ctx-menu__item" onClick={() => dispatch({ type: 'ADD_ROW' })}><Icon name="add" size={12} /> Insert Row</div>
          {ctxMenu.rowIdx !== undefined && (
            <>
              <div className="ds-ctx-menu__item" onClick={() => dispatch({ type: 'DUPLICATE_ROW', rowIdx: ctxMenu.rowIdx! })}><Icon name="content_copy" size={12} /> Duplicate Row</div>
              <div className="ds-ctx-menu__item ds-ctx-menu__item--danger" onClick={() => dispatch({ type: 'DELETE_ROW', rowIdx: ctxMenu.rowIdx! })}><Icon name="delete" size={12} /> Delete Row</div>
            </>
          )}
          <div className="ds-ctx-menu__sep" />
          {ctxMenu.col && (
            <div className="ds-ctx-menu__item ds-ctx-menu__item--danger" onClick={() => dispatch({ type: 'DELETE_COLUMN', name: ctxMenu.col! })}><Icon name="view_column" size={12} /> Delete Column</div>
          )}
          <div className="ds-ctx-menu__item" onClick={() => { const name = prompt('Column name?'); if(name) dispatch({ type: 'ADD_COLUMN', name }); }}><Icon name="view_column" size={12} /> Add Column</div>
        </div>
      )}
    </div>
  );
}

/* ─── Faker Types & Generator ─── */
const FAKER_TYPES = [
  { id: 'name', label: 'Full Name' },
  { id: 'email', label: 'Email' },
  { id: 'number', label: 'Integer' },
  { id: 'float', label: 'Decimal' },
  { id: 'date', label: 'Date' },
  { id: 'boolean', label: 'Boolean' },
  { id: 'city', label: 'City' },
  { id: 'country', label: 'Country' },
  { id: 'text', label: 'Lorem Text' },
] as const;
type FakerType = typeof FAKER_TYPES[number]['id'];

interface FakerColConfig {
  name: string;
  type: FakerType;
  options: any;
}

function generateFakerValue(type: FakerType, options: any = {}): any {
  const { min = 0, max = 1000, precision = 2, years = 5, words = 10, chance = 0.5 } = options;
  switch (type) {
    case 'name': return faker.person.fullName();
    case 'email': return faker.internet.email();
    case 'number': return faker.number.int({ min, max });
    case 'float': return parseFloat(faker.number.float({ min, max, fractionDigits: precision }).toFixed(precision));
    case 'date': return faker.date.past({ years }).toISOString().split('T')[0];
    case 'boolean': return Math.random() < chance;
    case 'city': return faker.location.city();
    case 'country': return faker.location.country();
    case 'text': return faker.lorem.words(words);
    default: return '';
  }
}

/* ─── Panels ─── */
/* ─── Panels ─── */
function TransformPanel() {
  const { state, addTransform } = useDataStudio();
  const { schema } = state;
  const [kind, setKind] = useState<TransformKind>('drop-column');
  const [col, setCol] = useState('');
  const [extra, setExtra] = useState<Record<string, string>>({});

  if (!schema) return null;

  const def = TRANSFORM_DEFS.find(d => d.kind === kind)!;
  const numericCols = schema.columns.filter(c => c.type === 'numeric').map(c => c.name);
  const allCols = schema.columns.map(c => c.name);

  const handleApply = () => {
    const params: Record<string, any> = {};
    if (def.needsCol) params.column = col || allCols[0];
    if (kind === 'rename-column') { params.from = col || allCols[0]; params.to = extra.newName || 'new_col'; }
    if (kind === 'fill-nulls') params.method = extra.method || 'mean';
    if (kind === 'fill-nulls' && extra.method === 'custom') params.value = extra.value || '';
    if (kind === 'bin-numeric') params.bins = parseInt(extra.bins || '5');
    if (kind === 'sort-by') params.ascending = extra.ascending !== 'false';
    if (kind === 'filter-rows') { params.op = extra.op || '=='; params.value = extra.value || ''; }
    const label = `${def.label}${def.needsCol ? ` [${params.column || params.from}]` : ''}`;
    addTransform(kind, params, label);
  };

  return (
    <div className="ds-transform-panel">
      <div className="ds-transform-panel__row">
        <div className="ds-field">
          <span className="ds-field__label">Operation</span>
          <select className="ds-field__select" value={kind} onChange={e => { setKind(e.target.value as TransformKind); setExtra({}); }}>
            {TRANSFORM_DEFS.map(d => <option key={d.kind} value={d.kind}>{d.label}</option>)}
          </select>
        </div>

        {def.needsCol && (
          <div className="ds-field">
            <span className="ds-field__label">Column</span>
            <select className="ds-field__select" value={col || allCols[0]} onChange={e => setCol(e.target.value)}>
              {((['min-max-scale', 'z-score', 'log-transform', 'clip-outliers', 'bin-numeric'].includes(kind)) ? numericCols : allCols)
                .map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
        )}
      </div>

      <div className="ds-transform-panel__row">
        {kind === 'rename-column' && (
          <div className="ds-field">
            <span className="ds-field__label">New Name</span>
            <input className="ds-field__input" value={extra.newName || ''} onChange={e => setExtra(p => ({ ...p, newName: e.target.value }))} placeholder="new_name" />
          </div>
        )}
        {kind === 'fill-nulls' && (
          <div className="ds-field">
            <span className="ds-field__label">Method</span>
            <select className="ds-field__select" value={extra.method || 'mean'} onChange={e => setExtra(p => ({ ...p, method: e.target.value }))}>
              {FILL_METHODS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
          </div>
        )}
        {kind === 'bin-numeric' && (
          <div className="ds-field">
            <span className="ds-field__label">Bins</span>
            <input className="ds-field__input" type="number" min={2} max={50} value={extra.bins || '5'} onChange={e => setExtra(p => ({ ...p, bins: e.target.value }))} />
          </div>
        )}
        {kind === 'filter-rows' && (
          <>
            <div className="ds-field">
              <span className="ds-field__label">Operator</span>
              <select className="ds-field__select" value={extra.op || '=='} onChange={e => setExtra(p => ({ ...p, op: e.target.value }))}>
                {FILTER_OPS.map(op => <option key={op} value={op}>{op}</option>)}
              </select>
            </div>
            <div className="ds-field">
              <span className="ds-field__label">Value</span>
              <input className="ds-field__input" value={extra.value || ''} onChange={e => setExtra(p => ({ ...p, value: e.target.value }))} />
            </div>
          </>
        )}
        <button className="ds-apply-btn" onClick={handleApply}>Apply</button>
      </div>
    </div>
  );
}

function SplitPanel() {
  const { state, dispatch, doSplit } = useDataStudio();
  const { splitConfig, split, schema } = state;
  if (!schema) return null;

  const downloadPart = (part: 'train' | 'test' | 'val') => {
    const data = split?.[part];
    if (!data) return;
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `split_${part}.csv`;
    link.click();
  };

  return (
    <div className="ds-split-panel">
      <div className="ds-transform-panel__row">
        <div className="ds-field">
          <span className="ds-field__label">Train ({Math.round(splitConfig.trainRatio * 100)}%)</span>
          <input className="ds-field__input" type="number" step="0.05" min="0" max="1" value={splitConfig.trainRatio} 
            onChange={e => dispatch({ type: 'SET_SPLIT_CONFIG', config: { trainRatio: Number(e.target.value) } })} style={{ width: 60 }} />
        </div>
        <div className="ds-field">
          <span className="ds-field__label">Test ({Math.round(splitConfig.testRatio * 100)}%)</span>
          <input className="ds-field__input" type="number" step="0.05" min="0" max="1" value={splitConfig.testRatio} 
            onChange={e => dispatch({ type: 'SET_SPLIT_CONFIG', config: { testRatio: Number(e.target.value) } })} style={{ width: 60 }} />
        </div>
        <div className="ds-field">
          <span className="ds-field__label">Val ({Math.round(splitConfig.valRatio * 100)}%)</span>
          <input className="ds-field__input" type="number" step="0.05" min="0" max="1" value={splitConfig.valRatio} 
            onChange={e => dispatch({ type: 'SET_SPLIT_CONFIG', config: { valRatio: Number(e.target.value) } })} style={{ width: 60 }} />
        </div>
        <button className="ds-apply-btn" onClick={doSplit} style={{ alignSelf: 'flex-end', height: 28 }}>Perform Split</button>
      </div>

      <div className="ds-split-panel__bar">
        <div className="ds-split-panel__bar-seg" style={{ width: `${splitConfig.trainRatio * 100}%`, background: 'var(--c-primary)' }} />
        <div className="ds-split-panel__bar-seg" style={{ width: `${splitConfig.testRatio * 100}%`, background: 'var(--c-tertiary)' }} />
        <div className="ds-split-panel__bar-seg" style={{ width: `${splitConfig.valRatio * 100}%`, background: '#ff8a80' }} />
      </div>

      {split && (
        <div className="ds-split-result">
          <div className="ds-split-result__chip">
            <span>TRAIN: <b>{split.train.length}</b></span>
            <button className="ds-split-result__dl" onClick={() => downloadPart('train')}><Icon name="download" size={12} /></button>
          </div>
          <div className="ds-split-result__chip">
            <span>TEST: <b>{split.test.length}</b></span>
            <button className="ds-split-result__dl" onClick={() => downloadPart('test')}><Icon name="download" size={12} /></button>
          </div>
          {split.val && (
            <div className="ds-split-result__chip">
              <span>VAL: <b>{split.val.length}</b></span>
              <button className="ds-split-result__dl" onClick={() => downloadPart('val')}><Icon name="download" size={12} /></button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SchemaView() {
  const { state, dispatch } = useDataStudio();
  const { schema, selectedColumn } = state;
  if (!schema) return null;

  return (
    <div style={{ padding: 12, overflowY: 'auto', flex: 1 }}>
      <table className="ds-schema-table">
        <thead>
          <tr><th>Column</th><th>Type</th><th>Nulls</th><th>Unique</th></tr>
        </thead>
        <tbody>
          {schema.columns.map(col => (
            <tr key={col.name} className={selectedColumn === col.name ? 'ds-schema-table__row--selected' : ''} onClick={() => dispatch({ type: 'SELECT_COLUMN', col: col.name })}>
              <td>{col.name}</td>
              <td><span className="ds-table__type-badge">{col.type}</span></td>
              <td>{col.nullCount}</td>
              <td>{col.uniqueCount}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VizArea() {
  const { state, dispatch } = useDataStudio();
  const { working, schema, columnStats, activeViz, selectedColumn } = state;
  const wrapRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({ w: 560, h: 380 });
  const [showSettings, setShowSettings] = useState(false);

  // Advanced Viz Config
  const [vizConfig, setVizConfig] = useState<{
    x: string; y: string; color: string; size: string;
    pointSize: number; opacity: number; colorScheme: 'default' | 'vibrant' | 'mono';
  }>({
    x: '', y: '', color: '', size: '',
    pointSize: 3, opacity: 0.6, colorScheme: 'default'
  });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) setContainerSize({ w: Math.floor(width), h: Math.floor(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  if (!schema) return null;
  const numericCols = schema.columns.filter(c => c.type === 'numeric').map(c => c.name);
  const allCols = schema.columns.map(c => c.name);

  // Sync defaults when schema/viz changes
  useEffect(() => {
    if (numericCols.length >= 2) {
      setVizConfig(prev => ({
        ...prev,
        x: prev.x || numericCols[0],
        y: prev.y || numericCols[1]
      }));
    }
  }, [numericCols.length]);

  const renderViz = () => {
    const W = containerSize.w - 32;
    const H = containerSize.h - (showSettings ? 120 : 16);
    if (working.length === 0) return <div className="ds-empty-state">No data to visualize</div>;

    switch (activeViz) {
      case 'histogram': {
        const col = selectedColumn && columnStats[selectedColumn]?.type === 'numeric' ? selectedColumn : numericCols[0];
        if (!col || !columnStats[col]) return <div className="ds-empty-state">Select a numeric column to view Histogram</div>;
        return <HistogramCanvas stats={columnStats[col].stats as NumericStats} width={W} height={H} label={col} />;
      }
      case 'bar-chart': {
        const catCols = schema.columns.filter(c => c.type === 'categorical').map(c => c.name);
        const col = selectedColumn && columnStats[selectedColumn]?.type === 'categorical' ? selectedColumn : catCols[0];
        if (!col || !columnStats[col]) return <div className="ds-empty-state">Select a categorical column to view Bar Chart</div>;
        const s = columnStats[col].stats as CategoricalStats;
        return (
          <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: 16 }}>
            <div style={{ flex: 1, minHeight: 0 }}>
              <BarChartCanvas labels={s.topN.map(t => t.value)} values={s.topN.map(t => t.count)} width={W} height={H - 120} title={col} />
            </div>
            <div className="ds-stats-summary" style={{ background: 'rgba(255,255,255,0.02)', borderRadius: 8, padding: 12, border: '1px solid rgba(255,255,255,0.04)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--c-primary)', textTransform: 'uppercase' }}>Frequency Summary</span>
                <span style={{ fontSize: 10, opacity: 0.5 }}>{s.uniqueCount} unique values</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))', gap: 8 }}>
                {s.topN.slice(0, 4).map((t, i) => (
                  <div key={i} style={{ background: 'rgba(255,255,255,0.03)', padding: '6px 10px', borderRadius: 4 }}>
                    <div style={{ fontSize: 9, opacity: 0.5, marginBottom: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{String(t.value)}</div>
                    <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
                      <span style={{ fontSize: 14, fontWeight: 800 }}>{t.count}</span>
                      <span style={{ fontSize: 9, opacity: 0.4 }}>({((t.count / working.length) * 100).toFixed(1)}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );
      }
      case 'scatter': {
        if (numericCols.length < 2) return <div className="ds-empty-state">Scatter plot requires at least 2 numeric columns</div>;
        const xCol = vizConfig.x || numericCols[0];
        const yCol = vizConfig.y || numericCols[1];
        return (
          <ScatterCanvas
            xData={working.map(r => Number(r[xCol]))}
            yData={working.map(r => Number(r[yCol]))}
            xLabel={xCol}
            yLabel={yCol}
            colorData={vizConfig.color ? working.map(r => r[vizConfig.color]) : undefined}
            sizeData={vizConfig.size ? working.map(r => Number(r[vizConfig.size])) : undefined}
            width={W}
            height={H}
            options={{ pointSize: vizConfig.pointSize, opacity: vizConfig.opacity, colorScheme: vizConfig.colorScheme }}
          />
        );
      }
      case 'correlation': {
        if (numericCols.length < 2) return <div className="ds-empty-state">Correlation Heatmap requires at least 2 numeric columns</div>;
        const { matrix, cols } = computeCorrelationMatrix(working, numericCols);
        return <CorrelationHeatmap cols={cols} matrix={matrix} width={W} height={H} />;
      }
      case 'box-plot': {
        const selectedStats = selectedColumn && columnStats[selectedColumn]?.type === 'numeric' 
          ? [columnStats[selectedColumn].stats as NumericStats] 
          : numericCols.map(c => columnStats[c]?.stats as NumericStats).filter(Boolean);
        const selectedLabels = selectedColumn && columnStats[selectedColumn]?.type === 'numeric'
          ? [selectedColumn]
          : numericCols;
        
        if (selectedStats.length === 0) return <div className="ds-empty-state">No numeric data for Box Plot</div>;
        return <BoxPlotCanvas stats={selectedStats} labels={selectedLabels} width={W} height={H} />;
      }
      case 'missing-map': {
        return <MissingMapCanvas data={working} columns={schema.columns.map(c => c.name)} width={W} height={H} />;
      }
      default: return <div>Viz "{activeViz}" coming soon...</div>;
    }
  };

  return (
    <div className="ds-viz-area">
      <div className="ds-viz-area__header">
        <div className="ds-view-toggle">
          {VIZ_OPTIONS.map(v => (
            <button key={v.id} className={`ds-view-toggle__btn ${activeViz === v.id ? 'ds-view-toggle__btn--active' : ''}`} onClick={() => dispatch({ type: 'SET_VIZ', viz: v.id })}>
              <Icon name={v.icon} size={12} /> {v.label}
            </button>
          ))}
        </div>
        <button className={`ds-faker-btn-opt ${showSettings ? 'ds-faker-btn-opt--active' : ''}`} onClick={() => setShowSettings(!showSettings)} style={{ marginLeft: 'auto', width: 'auto', padding: '0 10px', gap: 6 }}>
          <Icon name="tune" size={14} /> Options
        </button>
      </div>

      {showSettings && (
        <div className="ds-viz-settings">
          <div className="ds-viz-settings__grid">
            <div className="ds-field">
              <span className="ds-field__label">X Axis</span>
              <select className="ds-field__select" value={vizConfig.x} onChange={e => setVizConfig(p => ({ ...p, x: e.target.value }))}>
                {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="ds-field">
              <span className="ds-field__label">Y Axis</span>
              <select className="ds-field__select" value={vizConfig.y} onChange={e => setVizConfig(p => ({ ...p, y: e.target.value }))}>
                {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="ds-field">
              <span className="ds-field__label">Color (Z)</span>
              <select className="ds-field__select" value={vizConfig.color} onChange={e => setVizConfig(p => ({ ...p, color: e.target.value }))}>
                <option value="">— None —</option>
                {allCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="ds-field">
              <span className="ds-field__label">Size</span>
              <select className="ds-field__select" value={vizConfig.size} onChange={e => setVizConfig(p => ({ ...p, size: e.target.value }))}>
                <option value="">— Constant —</option>
                {numericCols.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
            <div className="ds-field">
              <span className="ds-field__label">Point Size ({vizConfig.pointSize})</span>
              <input type="range" min={1} max={10} step={0.5} value={vizConfig.pointSize} onChange={e => setVizConfig(p => ({ ...p, pointSize: parseFloat(e.target.value) }))} />
            </div>
            <div className="ds-field">
              <span className="ds-field__label">Opacity ({vizConfig.opacity})</span>
              <input type="range" min={0.1} max={1} step={0.1} value={vizConfig.opacity} onChange={e => setVizConfig(p => ({ ...p, opacity: parseFloat(e.target.value) }))} />
            </div>
          </div>
        </div>
      )}

      <div className="ds-viz-area__canvas-wrap" ref={wrapRef}>{renderViz()}</div>
    </div>
  );
}

function EditPanel({ showSynthetic, search }: { showSynthetic: boolean; search: any }) {
  const { state, dispatch } = useDataStudio();
  const { schema } = state;
  const [fakerCols, setFakerCols] = useState<FakerColConfig[]>([]);
  const [fakerRows, setFakerRows] = useState(100);
  const [appendMode, setAppendMode] = useState(true);
  const [expandedCol, setExpandedCol] = useState<number | null>(null);

  const getColsFromSchema = useCallback((): FakerColConfig[] => {
    if (!schema) return [];
    return schema.columns.map(c => {
      let type: FakerType = 'text';
      if (c.type === 'numeric') type = 'number';
      else if (c.type === 'boolean') type = 'boolean';
      else if (c.type === 'datetime') type = 'date';
      const name = c.name.toLowerCase();
      if (name.includes('name')) type = 'name';
      else if (name.includes('mail')) type = 'email';
      else if (name.includes('price') || name.includes('amount')) type = 'float';
      else if (name.includes('city')) type = 'city';
      else if (name.includes('country')) type = 'country';
      return { name: c.name, type, options: { min: 0, max: 1000, precision: 2, words: 5, years: 10 } };
    });
  }, [schema]);

  const syncWithSchema = useCallback(() => setFakerCols(getColsFromSchema()), [getColsFromSchema]);

  // Auto-load match schema on first mount if empty
  useEffect(() => {
    if (fakerCols.length === 0 && schema) syncWithSchema();
  }, [schema, syncWithSchema]);

  const generateData = () => {
    if (fakerCols.length === 0) return;
    dispatch({ type: 'SET_LOADING', active: true, message: `${appendMode ? 'Appending' : 'Generating'} ${fakerRows} rows…` });
    setTimeout(() => {
      const data: any[] = [];
      for (let i = 0; i < fakerRows; i++) {
        const row: any = {};
        fakerCols.forEach(col => { row[col.name] = generateFakerValue(col.type, col.options); });
        data.push(row);
      }
      if (appendMode) dispatch({ type: 'APPEND_DATA', data });
      else dispatch({ type: 'SET_WORKING_DATA', data, name: 'faker_dataset.csv', source: 'faker' });
    }, 100);
  };

  return (
    <div className="ds-faker-panel">
      <div className="ds-faker-panel__head">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className="ds-pipeline__label" style={{ fontSize: 11, fontWeight: 600, color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Data Editor</span>
          <div style={{ width: 1, height: 14, background: 'rgba(255,255,255,0.1)' }} />
          <button className="ds-apply-btn" onClick={() => dispatch({ type: 'ADD_ROW' })} style={{ background: 'rgba(207, 188, 255, 0.08)', color: 'var(--c-primary)', height: 26, fontSize: 10, padding: '0 12px', fontWeight: 600, border: '1px solid rgba(207, 188, 255, 0.1)' }}>
            <Icon name="add" size={14} /> ROW
          </button>
          <button className="ds-apply-btn" onClick={() => { const name = prompt('Column name?'); if(name) dispatch({ type: 'ADD_COLUMN', name }); }} style={{ background: 'rgba(207, 188, 255, 0.08)', color: 'var(--c-primary)', height: 26, fontSize: 10, padding: '0 12px', fontWeight: 600, border: '1px solid rgba(207, 188, 255, 0.1)' }}>
            <Icon name="view_column" size={14} /> COLUMN
          </button>
          <div style={{ width: 1, height: 14, background: 'rgba(255,255,255,0.1)' }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, background: 'rgba(255,255,255,0.03)', padding: '2px 8px', borderRadius: 4 }}>
            <Icon name="search" size={10} style={{ opacity: 0.4 }} />
            <input className="ds-field__input" placeholder="Find" value={search.find} onChange={e => search.setFind(e.target.value)} style={{ width: 60, height: 20, fontSize: 10, background: 'transparent', border: 'none' }} />
            <Icon name="arrow_forward" size={10} style={{ opacity: 0.4 }} />
            <input className="ds-field__input" placeholder="Replace" value={search.replace} onChange={e => search.setReplace(e.target.value)} style={{ width: 60, height: 20, fontSize: 10, background: 'transparent', border: 'none' }} />
            <button className="ds-apply-btn" onClick={() => dispatch({ type: 'BULK_REPLACE', find: search.find, replace: search.replace, column: null })} style={{ height: 18, fontSize: 8, padding: '0 6px', background: 'var(--c-primary)', color: 'var(--c-on-primary)' }}>Apply All</button>
            {search.find && <span style={{ fontSize: 8, opacity: 0.5, marginLeft: 4, fontStyle: 'italic' }}>{search.matchCount} matches</span>}
          </div>
        </div>
        {showSynthetic && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div className={`ds-faker-toggle ${appendMode ? 'ds-faker-toggle--active' : ''}`} onClick={() => setAppendMode(!appendMode)} title="Toggle append or replace mode">
              <div className="ds-faker-toggle__knob" />
              <div className="ds-faker-toggle__label">{appendMode ? 'APPEND' : 'REPLACE'}</div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(255,255,255,0.03)', padding: '0 8px', borderRadius: 4, height: 26 }}>
              <span style={{ fontSize: 9, opacity: 0.5, fontWeight: 700 }}>ROWS</span>
              <input className="ds-field__input" type="number" min={1} value={fakerRows} onChange={e => setFakerRows(Math.max(1, parseInt(e.target.value) || 1))} style={{ width: 45, border: 'none', background: 'transparent', height: 24, fontSize: 11, textAlign: 'center' }} />
            </div>
          </div>
        )}
      </div>
      {showSynthetic && (
        <>
          <div className="ds-faker-panel__cols">
        {fakerCols.map((col, i) => (
          <div key={i} className="ds-faker-item">
            <div className="ds-faker-panel__col-row" style={{ padding: '4px 10px', height: 32 }}>
              <Icon name="label" size={10} style={{ opacity: 0.3 }} />
              <input className="ds-field__input" value={col.name} onChange={e => setFakerCols(p => p.map((c, j) => j === i ? { ...c, name: e.target.value } : c))} style={{ flex: 1.2, fontWeight: 500, border: 'none', background: 'transparent', height: 24, fontSize: 12 }} placeholder="Field Name" />
              <div style={{ width: 1, height: 12, background: 'rgba(255,255,255,0.06)' }} />
              <select className="ds-field__select" value={col.type} onChange={e => setFakerCols(p => p.map((c, j) => j === i ? { ...c, type: e.target.value as FakerType } : c))} style={{ flex: 1, border: 'none', background: 'transparent', height: 24, fontSize: 11, color: 'rgba(255,255,255,0.6)' }}>
                {FAKER_TYPES.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
              </select>
              <button className={`ds-faker-btn-opt ${expandedCol === i ? 'ds-faker-btn-opt--active' : ''}`} onClick={() => setExpandedCol(expandedCol === i ? null : i)} title="Config" style={{ width: 24, height: 24 }}><Icon name="tune" size={12} /></button>
              <button className="ds-table__row-delete" style={{ position: 'static', opacity: 0.6, width: 24, height: 24 }} onClick={() => setFakerCols(p => p.filter((_, j) => j !== i))}><Icon name="close" size={12} /></button>
            </div>
            {expandedCol === i && (
              <div className="ds-faker-options">
                {['number', 'float'].includes(col.type) && (
                  <>
                    <div className="ds-field"><label className="ds-field__label">Min</label><input className="ds-field__input" type="number" value={col.options.min ?? 0} onChange={e => setFakerCols(p => p.map((c, j) => j === i ? { ...c, options: { ...c.options, min: Number(e.target.value) } } : c))} /></div>
                    <div className="ds-field"><label className="ds-field__label">Max</label><input className="ds-field__input" type="number" value={col.options.max ?? 1000} onChange={e => setFakerCols(p => p.map((c, j) => j === i ? { ...c, options: { ...c.options, max: Number(e.target.value) } } : c))} /></div>
                    {col.type === 'float' && <div className="ds-field"><label className="ds-field__label">Precision</label><input className="ds-field__input" type="number" value={col.options.precision ?? 2} onChange={e => setFakerCols(p => p.map((c, j) => j === i ? { ...c, options: { ...c.options, precision: Number(e.target.value) } } : c))} /></div>}
                  </>
                )}
                {col.type === 'text' && (
                  <div className="ds-field"><label className="ds-field__label">Words</label><input className="ds-field__input" type="number" value={col.options.words ?? 5} onChange={e => setFakerCols(p => p.map((c, j) => j === i ? { ...c, options: { ...c.options, words: Number(e.target.value) } } : c))} /></div>
                )}
                {col.type === 'date' && (
                  <div className="ds-field"><label className="ds-field__label">Years Past</label><input className="ds-field__input" type="number" value={col.options.years ?? 10} onChange={e => setFakerCols(p => p.map((c, j) => j === i ? { ...c, options: { ...c.options, years: Number(e.target.value) } } : c))} /></div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="ds-transform-panel__row" style={{ marginTop: 8, gap: 8, justifyContent: 'flex-start' }}>
        <button className="ds-apply-btn" onClick={() => setFakerCols([...fakerCols, { name: `col_${fakerCols.length + 1}`, type: 'text', options: { words: 5 } }])} style={{ background: 'rgba(255,255,255,0.04)', height: 26, fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '0 10px' }}>
          <Icon name="add" size={12} /> Field
        </button>
        <button className="ds-apply-btn" style={{ background: 'rgba(207, 188, 255, 0.08)', color: 'var(--c-primary)', height: 26, fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '0 10px' }} onClick={syncWithSchema}>
          <Icon name="auto_fix_high" size={12} /> Match
        </button>
        <button className="ds-apply-btn" style={{ background: 'rgba(255, 138, 128, 0.08)', color: '#ff8a80', height: 26, fontSize: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '0 10px' }} onClick={() => setFakerCols([])}>
          <Icon name="delete_sweep" size={12} /> Clear
        </button>
        <button className="ds-apply-btn" style={{ marginLeft: 'auto', background: 'var(--c-primary)', height: 28, fontSize: 11, fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '0 16px', borderRadius: 6 }} onClick={generateData} disabled={fakerCols.length === 0}>
          <Icon name="bolt" size={14} /> {appendMode ? 'Append' : 'Replace'} {fakerRows}
        </button>
      </div>
    </>
  )}
</div>
  );
}

export default function CenterPanel() {
  const { state, doExport } = useDataStudio();
  const { working, schema, activePanel } = state;
  const [centerView, setCenterView] = useState<'table' | 'viz'>('table');
  const [showSynthetic, setShowSynthetic] = useState(false);

  const [bulkFind, setBulkFind] = useState('');
  const [bulkReplace, setBulkReplace] = useState('');

  const matchCount = useMemo(() => {
    if (!bulkFind || working.length === 0) return 0;
    let count = 0;
    working.forEach(row => {
      Object.values(row).forEach(val => {
        if (String(val) === bulkFind) count++;
      });
    });
    return count;
  }, [bulkFind, working]);

  const searchState = { find: bulkFind, setFind: setBulkFind, replace: bulkReplace, setReplace: setBulkReplace, matchCount };

  if (working.length === 0) return <div className="ds-center"><Dropzone /></div>;

  return (
    <div className="ds-center">
      <div className="ds-toolbar">
        <div className="ds-toolbar__info">
          <span>ROWS: <span className="ds-toolbar__val">{working.length.toLocaleString()}</span></span>
          <span>COLS: <span className="ds-toolbar__val">{schema?.colCount ?? 0}</span></span>
        </div>
        <div className="ds-toolbar__actions">
          {activePanel === 'edit' && (
            <button className={`ds-toolbar__btn ${showSynthetic ? 'ds-toolbar__btn--active' : ''}`} onClick={() => setShowSynthetic(!showSynthetic)}>
              <Icon name="auto_awesome" size={14} /> Synthetic
            </button>
          )}
          <div className="ds-view-toggle">
            <button className={`ds-view-toggle__btn ${centerView === 'table' ? 'ds-view-toggle__btn--active' : ''}`} onClick={() => setCenterView('table')}>
              <Icon name="table_chart" size={12} /> Table
            </button>
            <button className={`ds-view-toggle__btn ${centerView === 'viz' ? 'ds-view-toggle__btn--active' : ''}`} onClick={() => setCenterView('viz')}>
              <Icon name="insights" size={12} /> Visualize
            </button>
          </div>
          <button className="ds-toolbar__btn ds-toolbar__btn--primary" onClick={doExport}><Icon name="download" size={14} /> Export</button>
        </div>
      </div>

      <div className="ds-subpanel" style={{ borderBottom: '1px solid var(--c-panel-border)', background: 'var(--c-surface-container-low)' }}>
        {(showSynthetic || activePanel === 'edit') && <EditPanel showSynthetic={showSynthetic} search={searchState} />}
        {activePanel === 'transform' && <TransformPanel />}
        {activePanel === 'split' && <SplitPanel />}
        {activePanel === 'schema' && <SchemaView />}
      </div>

      {centerView === 'table' ? <DataTable search={searchState} /> : <VizArea />}
    </div>
  );
}
