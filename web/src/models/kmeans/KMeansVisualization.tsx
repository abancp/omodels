import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClusteringData, trainKMeans, predictCluster, computeMetrics, computeDataStats, computeElbow, CLUSTER_COLORS, type Point, type KMeansState } from './math';
import { drawDataCanvas, drawInertiaCanvas, drawElbowCanvas } from './drawHelpers';

export default function KMeansVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const inertiaRef = useRef<HTMLCanvasElement>(null);
  const elbowRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [kmState, setKmState] = useState<KMeansState | null>(null);
  const [elbowData, setElbowData] = useState<{ k: number; inertia: number }[] | null>(null);
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);
  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = useState<{x: number; y: number; cluster: number; dist: number}[]>([]);

  const k = (params.k as number) ?? 3;
  const maxIter = (params.maxIter as number) ?? 50;
  const initMethod = (params.initMethod as string) ?? 'kmeans++';
  const showVoronoi = (params.showVoronoi as boolean) ?? true;
  const showCentroidPath = (params.showCentroidPath as boolean) ?? true;
  const elbowMaxK = (params.elbowMaxK as number) ?? 8;
  const numPoints = (datasetParams.points as number) ?? 150;
  const noise = (datasetParams.noise as number) ?? 1.0;

  const pushMetrics = useCallback((s: KMeansState | null) => {
    if (s) onMetricsUpdate(computeMetrics(s));
    else onMetricsUpdate([
      { label: 'Inertia', value: '—', isPrimary: true }, { label: 'Silhouette', value: '—' },
      { label: 'Iterations', value: '—' }, { label: 'K', value: '—' },
      { label: 'Converged', value: '—' }, { label: 'Clusters', value: '—' },
    ]);
  }, [onMetricsUpdate]);

  useEffect(() => { if (dataset === 'custom') return; setPoints(generateClusteringData(dataset, numPoints, noise)); setKmState(null); setElbowData(null); setInferResults([]); }, [dataset, numPoints, noise]);
  useEffect(() => { if (resetVersion === 0) return; setKmState(null); setElbowData(null); setInferResults([]); vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, [resetVersion]);

  // Mouse handlers
  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dragRef.current) return; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setPoints(prev => [...prev, { x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin) }]); }, []);
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => { if (!e.ctrlKey) return; e.preventDefault(); const f = e.deltaY > 0 ? 1.1 : 0.9; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; const mx = vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin); const my = vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin); vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dataset === 'custom') return; dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; }, [dataset]);
  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setHoverPt({ x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin), px: e.clientX - r.left, py: e.clientY - r.top }); if (!dragRef.current) return; const dr = dragRef.current; const dx = ((e.clientX - dr.sx) / r.width) * (dr.vp.xMax - dr.vp.xMin); const dy = ((e.clientY - dr.sy) / r.height) * (dr.vp.yMax - dr.vp.yMin); vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy }; setVpVer(v => v + 1); }, []);
  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);
  const zoomBtn = useCallback((f: number) => { const vp = vpRef.current; const mx = (vp.xMin + vp.xMax) / 2; const my = (vp.yMin + vp.yMax) / 2; vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);

  // Training
  useEffect(() => {
    if (!isTraining) return; if (points.length < k) { onTrainingComplete(); return; }
    let p = 0;
    const id = setInterval(() => { p += 0.10; if (p >= 1.0) { clearInterval(id);
      const s = trainKMeans(points, k, maxIter, initMethod);
      setKmState(s); pushMetrics(s);
      // Also compute elbow
      const eb = computeElbow(points, elbowMaxK, initMethod);
      setElbowData(eb);
      onTrainingComplete();
    }}, 50);
    return () => clearInterval(id);
  }, [isTraining]); // eslint-disable-line
  useEffect(() => { pushMetrics(kmState); }, [points]); // eslint-disable-line

  // Canvas renders
  useEffect(() => { const c = dataRef.current; if (!c) return; const r = () => drawDataCanvas(c, points, kmState, vpRef.current, inferResults, dataset, showVoronoi, showCentroidPath); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, kmState, vpVer, inferResults, dataset, showVoronoi, showCentroidPath]);
  useEffect(() => { const c = inertiaRef.current; if (!c) return; const r = () => drawInertiaCanvas(c, kmState); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [kmState]);
  useEffect(() => { const c = elbowRef.current; if (!c) return; const r = () => drawElbowCanvas(c, elbowData); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [elbowData]);

  const handleInfer = useCallback(() => {
    if (!kmState) return; const x = parseFloat(inferX), y = parseFloat(inferY); if (isNaN(x) || isNaN(y)) return;
    const cluster = predictCluster(x, y, kmState.centroids);
    const d = Math.sqrt((x - kmState.centroids[cluster].x) ** 2 + (y - kmState.centroids[cluster].y) ** 2);
    setInferResults(prev => [{ x, y, cluster, dist: d }, ...prev].slice(0, 5));
  }, [inferX, inferY, kmState]);

  const stats = useMemo(() => computeDataStats(points), [points]);

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER/CLUSTER MAP */}
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

      {/* 2. INERTIA CURVE */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={inertiaRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. ELBOW + CLUSTER INFO */}
      <div className="viz-scroll__section viz-scroll__section--controls" style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">ELBOW METHOD</span></div>
          <div style={{ height: '120px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)', overflow: 'hidden' }}>
            <canvas ref={elbowRef} style={{ width: '100%', height: '100%', display: 'block' }} />
          </div>
        </div>
        {kmState && (
          <div style={{ flex: 1 }}>
            <div className="viz-ctrl__header"><span className="viz-ctrl__title">CLUSTER SIZES</span></div>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {kmState.clusterSizes.map((size, i) => (
                <div key={i} style={{ padding: '8px 12px', borderRadius: '6px', background: `${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}18`, border: `1px solid ${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}40`, textAlign: 'center', minWidth: '60px' }}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', color: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}>{size}</div>
                  <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)' }}>C{i + 1}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 4. CLUSTER INFO */}
      {kmState && (
        <div className="viz-scroll__section viz-scroll__section--controls">
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">CLUSTERING INFO</span><span className="viz-ctrl__subtitle">K={kmState.k}, init={initMethod}</span></div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-primary)' }}>Convergence</h4>
              <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Iterations:</span><span>{kmState.iteration}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Converged:</span><span style={{ color: kmState.converged ? '#4ade80' : '#f87171' }}>{kmState.converged ? 'Yes' : 'No'}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Initial Inertia:</span><span>{kmState.inertiaHistory[0]?.toFixed(3) ?? '—'}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Final Inertia:</span><span style={{ color: 'var(--c-primary)' }}>{kmState.inertia.toFixed(3)}</span></div>
              </div>
            </div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-tertiary)' }}>Centroids</h4>
              <div style={{ fontSize: '10px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '3px' }}>
                {kmState.centroids.map((c, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ width: '8px', height: '8px', borderRadius: '2px', background: CLUSTER_COLORS[i % CLUSTER_COLORS.length], display: 'inline-block' }} />
                      C{i + 1}
                    </span>
                    <span>({c.x.toFixed(3)}, {c.y.toFixed(3)})</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          {/* Silhouette bar */}
          <div style={{ marginTop: '12px', padding: '10px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
              <span style={{ fontSize: '10px', fontWeight: 600, color: 'var(--c-on-surface-variant)', letterSpacing: '0.5px' }}>SILHOUETTE SCORE</span>
              <span style={{ fontSize: '12px', fontWeight: 700, color: kmState.silhouetteScore > 0.5 ? '#4ade80' : kmState.silhouetteScore > 0.25 ? '#fbbf24' : '#f87171' }}>{kmState.silhouetteScore.toFixed(3)}</span>
            </div>
            <div style={{ height: '6px', background: 'var(--c-surface-variant)', borderRadius: '3px', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${Math.max(0, (kmState.silhouetteScore + 1) / 2 * 100)}%`, background: kmState.silhouetteScore > 0.5 ? '#4ade80' : kmState.silhouetteScore > 0.25 ? '#fbbf24' : '#f87171', borderRadius: '3px', transition: 'width 0.3s' }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '8px', color: 'var(--c-on-surface-variant)', opacity: 0.5, marginTop: '2px' }}><span>-1</span><span>0</span><span>+1</span></div>
          </div>
        </div>
      )}

      {/* 5. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header"><span className="viz-ctrl__title">INFERENCE</span><span className="viz-ctrl__subtitle">Nearest centroid assignment</span></div>
        <div className="viz-infer__input-row">
          <div className="viz-infer__field"><label>Input x₁</label><input className="viz-infer__input" type="number" step="0.01" value={inferX} onChange={e => setInferX(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <div className="viz-infer__field"><label>Input x₂</label><input className="viz-infer__input" type="number" step="0.01" value={inferY} onChange={e => setInferY(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <button className="viz-infer__btn" onClick={handleInfer} disabled={!kmState}>Predict</button>
          <div className="viz-infer__result" style={{ marginLeft: '10px' }}><label>Cluster</label><span className="viz-infer__y" style={{ color: inferResults.length > 0 ? CLUSTER_COLORS[inferResults[0].cluster % CLUSTER_COLORS.length] : 'inherit' }}>{inferResults.length > 0 ? `C${inferResults[0].cluster + 1}` : '—'}</span></div>
          <div className="viz-infer__result"><label>Distance</label><span className="viz-infer__y">{inferResults.length > 0 ? inferResults[0].dist.toFixed(4) : '—'}</span></div>
        </div>
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr' }}><span>x₁</span><span>x₂</span><span>Cluster</span><span>Dist</span></div>
            {inferResults.map((r, i) => (
              <div key={i} className="viz-infer__history-row" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr' }}>
                <span>{r.x.toFixed(3)}</span><span>{r.y.toFixed(3)}</span>
                <span style={{ color: CLUSTER_COLORS[r.cluster % CLUSTER_COLORS.length], fontWeight: 'bold' }}>C{r.cluster + 1}</span>
                <span>{r.dist.toFixed(4)}</span>
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
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₁</span><span className="viz-stats__val">{stats.meanX.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₂</span><span className="viz-stats__val">{stats.meanY.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Std x₁</span><span className="viz-stats__val">{stats.stdX.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Std x₂</span><span className="viz-stats__val">{stats.stdY.toFixed(3)}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
