import { useRef, useEffect, useState, useCallback, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClusteringData, type Point, initGMM, gmmStep, predictGMM, computeMetrics, type GMMState } from './math';
import { drawDataCanvas, drawLikelihoodCanvas, CLUSTER_COLORS } from './drawHelpers';

export default function GMMVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const llRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [gmmState, setGmmState] = useState<GMMState | null>(null);
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);
  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = useState<{x: number; y: number; probs: number[]}[]>([]);

  const k = (params.k as number) ?? 3;
  const maxIter = (params.maxIter as number) ?? 100;
  const covType = (params.covarianceType as string) ?? 'full';
  const initMethod = (params.initMethod as string) ?? 'kmeans++';
  const showCovariance = (params.showCovariance as boolean) ?? true;
  const colorMixing = (params.colorMixing as boolean) ?? true;
  
  const numPoints = (datasetParams.points as number) ?? 200;
  const noise = (datasetParams.noise as number) ?? 1.0;

  const pushMetrics = useCallback((s: GMMState | null) => {
    if (s) onMetricsUpdate(computeMetrics(s));
    else onMetricsUpdate([
      { label: 'Log-Likelihood', value: '—', isPrimary: true }, { label: 'Iterations', value: '—' },
      { label: 'Converged', value: '—' }, { label: 'AIC', value: '—' }
    ]);
  }, [onMetricsUpdate]);

  useEffect(() => { if (dataset === 'custom') return; setPoints(generateClusteringData(dataset, numPoints, noise)); setGmmState(null); setInferResults([]); }, [dataset, numPoints, noise]);
  useEffect(() => { if (resetVersion === 0) return; setGmmState(null); setInferResults([]); vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, [resetVersion]);
  useEffect(() => { setGmmState(null); }, [k, covType, initMethod]); // Reset on param changes

  // Mouse handlers
  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dragRef.current) return; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setPoints(prev => [...prev, { x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin) }]); setGmmState(null); }, []);
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => { if (!e.ctrlKey) return; e.preventDefault(); const f = e.deltaY > 0 ? 1.1 : 0.9; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; const mx = vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin); const my = vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin); vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dataset === 'custom') return; dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; }, [dataset]);
  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setHoverPt({ x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin), px: e.clientX - r.left, py: e.clientY - r.top }); if (!dragRef.current) return; const dr = dragRef.current; const dx = ((e.clientX - dr.sx) / r.width) * (dr.vp.xMax - dr.vp.xMin); const dy = ((e.clientY - dr.sy) / r.height) * (dr.vp.yMax - dr.vp.yMin); vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy }; setVpVer(v => v + 1); }, []);
  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);
  const zoomBtn = useCallback((f: number) => { const vp = vpRef.current; const mx = (vp.xMin + vp.xMax) / 2; const my = (vp.yMin + vp.yMax) / 2; vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);

  // Training
  useEffect(() => {
    if (!isTraining) return;
    if (points.length < k) { onTrainingComplete(); return; }
    
    let state = gmmState || initGMM(points, k, covType, initMethod);
    
    const id = setInterval(() => {
      state = gmmStep(points, state);
      setGmmState(state);
      
      if (state.converged || state.iteration >= maxIter) {
        clearInterval(id);
        onTrainingComplete();
      }
    }, 50);
    
    return () => clearInterval(id);
  }, [isTraining]); // eslint-disable-line
  
  useEffect(() => { pushMetrics(gmmState); }, [gmmState, pushMetrics]);

  // Canvas Renders
  useEffect(() => { const c = dataRef.current; if (!c) return; const r = () => drawDataCanvas(c, points, gmmState, vpRef.current, inferResults, dataset, showCovariance, colorMixing); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, gmmState, vpVer, inferResults, dataset, showCovariance, colorMixing]);
  useEffect(() => { const c = llRef.current; if (!c) return; const r = () => drawLikelihoodCanvas(c, gmmState); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [gmmState]);

  const handleInfer = useCallback(() => {
    if (!gmmState || gmmState.iteration === 0) return;
    const x = parseFloat(inferX), y = parseFloat(inferY); if (isNaN(x) || isNaN(y)) return;
    const probs = predictGMM({ x, y }, gmmState);
    setInferResults(prev => [{ x, y, probs }, ...prev].slice(0, 5));
  }, [inferX, inferY, gmmState]);

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER MAP */}
      <div className="viz-scroll__section viz-scroll__section--canvas" style={{ position: 'relative' }}>
        <canvas ref={dataRef} onContextMenu={e => e.preventDefault()} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={() => { handleMouseUp(); setHoverPt(null); }} onClick={dataset === 'custom' ? handleDataClick : undefined} onWheel={handleWheel} style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }} />
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

      {/* 2. LOG-LIKELIHOOD CURVE */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={llRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. GAUSSIAN MIXTURE INFO */}
      {gmmState && gmmState.iteration > 0 && (
        <div className="viz-scroll__section viz-scroll__section--controls">
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">COMPONENTS</span><span className="viz-ctrl__subtitle">K={gmmState.k}, Type={covType}</span></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '8px' }}>
            {gmmState.means.map((mean, i) => (
              <div key={i} style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)', borderLeft: `3px solid ${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ fontSize: '11px', fontWeight: 'bold', color: 'var(--c-on-surface)' }}>Component {i + 1}</span>
                  <span style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)' }}>{(gmmState.weights[i] * 100).toFixed(1)}%</span>
                </div>
                <div style={{ fontSize: '10px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '2px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>μ:</span><span>({mean.x.toFixed(2)}, {mean.y.toFixed(2)})</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>σ²:</span>
                    <span>[{gmmState.covariances[i].vxx.toFixed(3)}, {gmmState.covariances[i].vyy.toFixed(3)}]</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 4. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header"><span className="viz-ctrl__title">INFERENCE</span><span className="viz-ctrl__subtitle">Soft responsibility prediction</span></div>
        <div className="viz-infer__input-row">
          <div className="viz-infer__field"><label>Input x₁</label><input className="viz-infer__input" type="number" step="0.01" value={inferX} onChange={e => setInferX(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <div className="viz-infer__field"><label>Input x₂</label><input className="viz-infer__input" type="number" step="0.01" value={inferY} onChange={e => setInferY(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <button className="viz-infer__btn" onClick={handleInfer} disabled={!gmmState || gmmState.iteration === 0}>Predict</button>
        </div>
        
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            {inferResults.map((r, i) => (
              <div key={i} style={{ padding: '8px 0', borderBottom: i < inferResults.length - 1 ? '1px solid var(--c-panel-border)' : 'none' }}>
                <div style={{ fontSize: '10px', color: 'var(--c-on-surface-variant)', marginBottom: '4px' }}>x=({r.x.toFixed(2)}, {r.y.toFixed(2)})</div>
                <div style={{ display: 'flex', gap: '2px', height: '16px', borderRadius: '4px', overflow: 'hidden' }}>
                  {r.probs.map((p, j) => p > 0.01 && (
                    <div key={j} style={{ width: `${p * 100}%`, background: CLUSTER_COLORS[j % CLUSTER_COLORS.length], display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '9px', color: '#000', fontWeight: 'bold' }}>
                      {p > 0.1 ? `${(p * 100).toFixed(0)}%` : ''}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
