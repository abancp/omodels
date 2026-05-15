import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClassificationData, trainRandomForest, rfPredictSingle, rfPredictProbability, computeMetrics, computeConfusionMatrix, computeDataStats, getTreeVotes, type Point, type RFState } from './math';
import { drawDataCanvas, drawTreeGrid, drawROCCanvas } from './drawHelpers';

export default function RandomForestVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const gridRef = useRef<HTMLCanvasElement>(null);
  const rocRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [rfState, setRfState] = useState<RFState | null>(null);
  const [selectedTree, setSelectedTree] = useState(0);
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);
  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = useState<{x: number; y: number; prob: number; cls: number; votes?: {v0: number; v1: number}}[]>([]);

  const nEstimators = (params.nEstimators as number) ?? 10;
  const maxDepth = (params.maxDepth as number) ?? 5;
  const maxFeatures = (params.maxFeatures as string) ?? 'sqrt';
  const algorithm = (params.algorithm as 'id3'|'c45'|'cart') ?? 'cart';
  const minSamplesSplit = (params.minSamplesSplit as number) ?? 2;
  const minSamplesLeaf = (params.minSamplesLeaf as number) ?? 1;
  const numBins = (params.numBins as number) ?? 15;
  const showTreeVotes = (params.showTreeVotes as boolean) ?? true;
  const numPoints = (datasetParams.points as number) ?? 150;
  const noise = (datasetParams.noise as number) ?? 0.2;

  const pushMetrics = useCallback((s: RFState | null) => {
    if (s) onMetricsUpdate(computeMetrics(points, s));
    else onMetricsUpdate([
      { label: 'Accuracy', value: '—', isPrimary: true }, { label: 'Precision', value: '—' },
      { label: 'Recall', value: '—' }, { label: 'F1 Score', value: '—' },
      { label: 'OOB Accuracy', value: '—' }, { label: 'Trees', value: '—' },
    ]);
  }, [points, onMetricsUpdate]);

  useEffect(() => { if (dataset === 'custom') return; setPoints(generateClassificationData(dataset, numPoints, noise)); setRfState(null); setInferResults([]); }, [dataset, numPoints, noise]);
  useEffect(() => { if (resetVersion === 0) return; setRfState(null); setInferResults([]); setSelectedTree(0); vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, [resetVersion]);

  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current) return; const canvas = dataRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect(); const vp = vpRef.current;
    const px = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    setPoints(prev => [...prev, { x: px, y: py, cls: e.shiftKey ? 1 : 0 }]);
  }, []);
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    if (!e.ctrlKey) return; e.preventDefault(); const f = e.deltaY > 0 ? 1.1 : 0.9;
    const canvas = dataRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect(); const vp = vpRef.current;
    const mx = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const my = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f };
    setVpVer(v => v + 1);
  }, []);
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dataset === 'custom') return; dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; }, [dataset]);
  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    const canvas = dataRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect(); const vp = vpRef.current;
    setHoverPt({ x: vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin), px: e.clientX - rect.left, py: e.clientY - rect.top });
    if (!dragRef.current) return; const dr = dragRef.current;
    const dx = ((e.clientX - dr.sx) / rect.width) * (dr.vp.xMax - dr.vp.xMin);
    const dy = ((e.clientY - dr.sy) / rect.height) * (dr.vp.yMax - dr.vp.yMin);
    vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy }; setVpVer(v => v + 1);
  }, []);
  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);
  const zoomBtn = useCallback((f: number) => { const vp = vpRef.current; const mx = (vp.xMin + vp.xMax) / 2; const my = (vp.yMin + vp.yMax) / 2; vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);

  // Training with animated progress
  useEffect(() => {
    if (!isTraining) return;
    if (points.length < 2) { onTrainingComplete(); return; }
    let progress = 0;
    const id = setInterval(() => {
      progress += 0.12;
      if (progress >= 1.0) {
        clearInterval(id);
        const s = trainRandomForest(points, nEstimators, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, algorithm, numBins);
        setRfState(s); pushMetrics(s); onTrainingComplete();
      }
    }, 50);
    return () => clearInterval(id);
  }, [isTraining]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { pushMetrics(rfState); }, [points]); // eslint-disable-line

  // Canvas renders
  useEffect(() => { const c = dataRef.current; if (!c) return; const r = () => drawDataCanvas(c, points, rfState, vpRef.current, inferResults, dataset); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, rfState, vpVer, inferResults, dataset]);
  useEffect(() => { const c = gridRef.current; if (!c) return; const r = () => drawTreeGrid(c, rfState, vpRef.current, selectedTree); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [rfState, vpVer, selectedTree]);
  useEffect(() => { const c = rocRef.current; if (!c) return; const r = () => drawROCCanvas(c, points, rfState); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, rfState]);

  const handleInfer = useCallback(() => {
    if (!rfState) return; const x = parseFloat(inferX), y = parseFloat(inferY); if (isNaN(x) || isNaN(y)) return;
    const cls = rfPredictSingle(x, y, rfState); const prob = rfPredictProbability(x, y, rfState);
    const votes = getTreeVotes(x, y, rfState);
    const v0 = votes.filter(v => v.vote === 0).length, v1 = votes.filter(v => v.vote === 1).length;
    setInferResults(prev => [{ x, y, prob, cls, votes: { v0, v1 } }, ...prev].slice(0, 5));
  }, [inferX, inferY, rfState]);

  const stats = useMemo(() => computeDataStats(points), [points]);
  const cm = useMemo(() => rfState ? computeConfusionMatrix(points, rfState) : { tp: 0, tn: 0, fp: 0, fn: 0 }, [points, rfState]);
  // Tree votes for last inference
  const lastVotes = useMemo(() => { if (!rfState || inferResults.length === 0) return null; const r = inferResults[0]; return getTreeVotes(r.x, r.y, rfState); }, [rfState, inferResults]);

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER */}
      <div className="viz-scroll__section viz-scroll__section--canvas" style={{ position: 'relative' }}>
        <canvas ref={dataRef} onContextMenu={e => e.preventDefault()} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onClick={dataset === 'custom' ? handleDataClick : undefined} onWheel={handleWheel} style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }} />
        {hoverPt && <div className="viz-tooltip" style={{ left: hoverPt.px + 10, top: hoverPt.py - 24 }}>{hoverPt.x.toFixed(2)}, {hoverPt.y.toFixed(2)}</div>}
        <div className="viz-scatter-ctrls">
          <button className="viz-scatter-btn" onClick={resetView} title="Reset view">⟲</button>
          <button className="viz-scatter-btn" onClick={() => zoomBtn(0.8)} title="Zoom In">+</button>
          <button className="viz-scatter-btn" onClick={() => zoomBtn(1.2)} title="Zoom Out">−</button>
          <button className="viz-scatter-btn" onClick={(e) => {
            const container = (e.target as HTMLElement).closest('.viz-scroll__section--canvas');
            if (container) {
              if (document.fullscreenElement) {
                document.exitFullscreen();
              } else {
                container.requestFullscreen();
              }
            }
          }} title="Full Screen">⛶</button>
        </div>
      </div>

      {/* 2. INDIVIDUAL TREE BOUNDARIES */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={gridRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. CONFUSION MATRIX & ROC */}
      <div className="viz-scroll__section viz-scroll__section--controls" style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">CONFUSION MATRIX</span></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: '4px', textAlign: 'center', fontSize: '11px' }}>
            <div></div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 0</div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 1</div>
            <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 0</div>
            <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '14px' }}>{cm.tn}</div><div>TN</div></div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{cm.fp}</div><div>FP</div></div>
            <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 1</div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{cm.fn}</div><div>FN</div></div>
            <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '14px' }}>{cm.tp}</div><div>TP</div></div>
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">ROC CURVE</span></div>
          <div style={{ height: '120px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)', overflow: 'hidden' }}>
            <canvas ref={rocRef} style={{ width: '100%', height: '100%', display: 'block' }} />
          </div>
        </div>
      </div>

      {/* 4. FOREST INFO */}
      {rfState && (
        <div className="viz-scroll__section viz-scroll__section--controls">
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">FOREST INFO</span><span className="viz-ctrl__subtitle">Ensemble of {rfState.nEstimators} trees</span></div>
          <div className="viz-ctrl__split">
            <div className="viz-ctrl__right" style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-primary)' }}>Ensemble</h4>
                <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Trees:</span><span>{rfState.nEstimators}</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Max Depth:</span><span>{rfState.maxDepth}</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Max Features:</span><span>{rfState.maxFeatures}</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>OOB Error:</span><span style={{ color: 'var(--c-error)' }}>{(rfState.oobError * 100).toFixed(1)}%</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>OOB Accuracy:</span><span style={{ color: 'var(--c-primary)' }}>{(rfState.oobAccuracy * 100).toFixed(1)}%</span></div>
                </div>
              </div>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-tertiary)' }}>Feature Importance</h4>
                <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '8px' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}><span>Feature x₁:</span><span>{(rfState.featureImportance.x * 100).toFixed(1)}%</span></div>
                    <div style={{ height: '4px', background: 'var(--c-surface-variant)', borderRadius: '2px', overflow: 'hidden' }}><div style={{ height: '100%', width: `${rfState.featureImportance.x * 100}%`, background: 'var(--c-primary)', borderRadius: '2px' }} /></div>
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}><span>Feature x₂:</span><span>{(rfState.featureImportance.y * 100).toFixed(1)}%</span></div>
                    <div style={{ height: '4px', background: 'var(--c-surface-variant)', borderRadius: '2px', overflow: 'hidden' }}><div style={{ height: '100%', width: `${rfState.featureImportance.y * 100}%`, background: 'var(--c-tertiary)', borderRadius: '2px' }} /></div>
                  </div>
                  <div style={{ marginTop: '4px', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Avg Nodes/Tree:</span>
                    <span>{(rfState.trees.reduce((s, t) => s + t.nodeCount, 0) / rfState.trees.length).toFixed(1)}</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Avg Depth:</span>
                    <span>{(rfState.trees.reduce((s, t) => s + t.treeDepth, 0) / rfState.trees.length).toFixed(1)}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          {/* Tree selector */}
          <div style={{ marginTop: '12px' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--c-on-surface-variant)', marginBottom: '6px', letterSpacing: '0.5px' }}>SELECT TREE</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
              {rfState.trees.map((t, i) => (
                <button key={i} onClick={() => setSelectedTree(i)} style={{
                  padding: '3px 8px', borderRadius: '4px', fontSize: '10px', cursor: 'pointer',
                  background: i === selectedTree ? 'var(--c-primary)' : 'var(--c-surface-variant)',
                  color: i === selectedTree ? '#000' : 'var(--c-on-surface-variant)',
                  border: '1px solid var(--c-panel-border)', fontWeight: i === selectedTree ? 'bold' : 'normal',
                }}>T{i + 1} <span style={{ opacity: 0.7 }}>({t.nodeCount}n)</span></button>
              ))}
            </div>
            {rfState.trees[selectedTree] && (
              <div style={{ marginTop: '8px', padding: '8px', background: 'var(--c-surface-variant)', borderRadius: '4px', fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px' }}>
                <div><span style={{ opacity: 0.6 }}>Nodes: </span>{rfState.trees[selectedTree].nodeCount}</div>
                <div><span style={{ opacity: 0.6 }}>Leaves: </span>{rfState.trees[selectedTree].leafCount}</div>
                <div><span style={{ opacity: 0.6 }}>Depth: </span>{rfState.trees[selectedTree].treeDepth}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 5. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header"><span className="viz-ctrl__title">INFERENCE</span><span className="viz-ctrl__subtitle">Ensemble prediction with voting</span></div>
        <div className="viz-infer__input-row">
          <div className="viz-infer__field"><label>Input x₁</label><input className="viz-infer__input" type="number" step="0.01" value={inferX} onChange={e => setInferX(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <div className="viz-infer__field"><label>Input x₂</label><input className="viz-infer__input" type="number" step="0.01" value={inferY} onChange={e => setInferY(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <button className="viz-infer__btn" onClick={handleInfer} disabled={!rfState}>Predict</button>
          <div className="viz-infer__result" style={{ marginLeft: '10px' }}><label>P(y=1)</label><span className="viz-infer__y">{inferResults.length > 0 ? inferResults[0].prob.toFixed(4) : '—'}</span></div>
          <div className="viz-infer__result"><label>Class</label><span className="viz-infer__y" style={{ color: inferResults.length > 0 ? (inferResults[0].cls === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)') : 'inherit' }}>{inferResults.length > 0 ? inferResults[0].cls : '—'}</span></div>
        </div>
        {/* Vote breakdown */}
        {showTreeVotes && lastVotes && inferResults.length > 0 && (
          <div style={{ marginTop: '12px', padding: '10px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--c-on-surface-variant)', marginBottom: '8px', letterSpacing: '0.5px' }}>TREE VOTES ({inferResults[0].votes?.v0 ?? 0} vs {inferResults[0].votes?.v1 ?? 0})</div>
            <div style={{ display: 'flex', gap: '2px', height: '20px', borderRadius: '4px', overflow: 'hidden', marginBottom: '6px' }}>
              <div style={{ width: `${((inferResults[0].votes?.v0 ?? 0) / (rfState?.nEstimators ?? 1)) * 100}%`, background: 'var(--c-primary)', transition: 'width 0.3s' }} />
              <div style={{ width: `${((inferResults[0].votes?.v1 ?? 0) / (rfState?.nEstimators ?? 1)) * 100}%`, background: 'var(--c-tertiary)', transition: 'width 0.3s' }} />
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px' }}>
              {lastVotes.map((v, i) => (
                <div key={i} style={{ width: '18px', height: '18px', borderRadius: '3px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '8px', fontWeight: 'bold', background: v.vote === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)', color: '#000' }}>{i + 1}</div>
              ))}
            </div>
          </div>
        )}
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr' }}><span>x₁</span><span>x₂</span><span>P(y=1)</span><span>Class</span><span>Votes</span></div>
            {inferResults.map((r, i) => (
              <div key={i} className="viz-infer__history-row" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr' }}>
                <span>{r.x.toFixed(3)}</span><span>{r.y.toFixed(3)}</span><span>{r.prob.toFixed(4)}</span>
                <span style={{ color: r.cls === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)' }}>{r.cls}</span>
                <span>{r.votes?.v0 ?? 0}/{r.votes?.v1 ?? 0}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 6. DATA STATISTICS */}
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">DATA STATISTICS</span></div>
          <div className="viz-stats__grid">
            <div className="viz-stats__item"><span className="viz-stats__label">N</span><span className="viz-stats__val">{stats.n}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 0</span><span className="viz-stats__val viz-stats__val--primary">{stats.nClass0}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 1</span><span className="viz-stats__val" style={{ color: 'var(--c-tertiary)' }}>{stats.nClass1}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₁</span><span className="viz-stats__val">{stats.meanX.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₂</span><span className="viz-stats__val">{stats.meanY.toFixed(3)}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
