import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClusteringData, type Point, initDBSCAN, dbscanStep, computeMetrics, type DBSCANState } from './math';
import { drawDataCanvas, CLUSTER_COLORS } from './drawHelpers';

export default function DBSCANVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [dbState, setDbState] = useState<DBSCANState | null>(null);
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);

  const eps = (params.eps as number) ?? 0.1;
  const minPts = (params.minPts as number) ?? 4;
  const metric = (params.metric as string) ?? 'euclidean';
  const showEpsCircles = (params.showEpsCircles as boolean) ?? true;
  const showPointTypes = (params.showPointTypes as boolean) ?? true;
  
  const numPoints = (datasetParams.points as number) ?? 300;
  const noise = (datasetParams.noise as number) ?? 0.05;

  const pushMetrics = useCallback((s: DBSCANState | null) => {
    if (s) onMetricsUpdate(computeMetrics(s, points.length));
    else onMetricsUpdate([
      { label: 'Clusters', value: '—', isPrimary: true }, { label: 'Core Pts', value: '—' },
      { label: 'Border Pts', value: '—' }, { label: 'Noise Pts', value: '—' }, { label: '% Noise', value: '—' }
    ]);
  }, [onMetricsUpdate, points.length]);

  useEffect(() => { if (dataset === 'custom') return; setPoints(generateClusteringData(dataset, numPoints, noise)); setDbState(null); }, [dataset, numPoints, noise]);
  useEffect(() => { if (resetVersion === 0) return; setDbState(null); vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, [resetVersion]);
  useEffect(() => { setDbState(null); }, [eps, minPts, metric]); // reset if params change
  
  // Mouse handlers
  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dragRef.current) return; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setPoints(prev => [...prev, { x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin) }]); setDbState(null); }, []);
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => { if (!e.ctrlKey) return; e.preventDefault(); const f = e.deltaY > 0 ? 1.1 : 0.9; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; const mx = vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin); const my = vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin); vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dataset === 'custom') return; dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; }, [dataset]);
  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setHoverPt({ x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin), px: e.clientX - r.left, py: e.clientY - r.top }); if (!dragRef.current) return; const dr = dragRef.current; const dx = ((e.clientX - dr.sx) / r.width) * (dr.vp.xMax - dr.vp.xMin); const dy = ((e.clientY - dr.sy) / r.height) * (dr.vp.yMax - dr.vp.yMin); vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy }; setVpVer(v => v + 1); }, []);
  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);
  const zoomBtn = useCallback((f: number) => { const vp = vpRef.current; const mx = (vp.xMin + vp.xMax) / 2; const my = (vp.yMin + vp.yMax) / 2; vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);

  // Training
  useEffect(() => {
    if (!isTraining) return;
    if (points.length === 0) { onTrainingComplete(); return; }
    
    let state = dbState || initDBSCAN(points.length, eps, minPts, metric);
    
    const id = setInterval(() => {
      // Small step size to create a visual "growth" animation
      state = dbscanStep(points, state, 1);
      setDbState(state);
      if (state.phase === 'DONE') {
        clearInterval(id);
        onTrainingComplete();
      }
    }, 15);
    
    return () => clearInterval(id);
  }, [isTraining]); // eslint-disable-line
  
  useEffect(() => { pushMetrics(dbState); }, [dbState, pushMetrics]);

  // Canvas render
  useEffect(() => { const c = dataRef.current; if (!c) return; const r = () => drawDataCanvas(c, points, dbState, vpRef.current, hoverPt, dataset, showEpsCircles, showPointTypes); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, dbState, vpVer, hoverPt, dataset, showEpsCircles, showPointTypes]);

  // Compute cluster sizes
  const clusterSizes = useMemo(() => {
    if (!dbState || dbState.phase !== 'DONE') return [];
    const sizes: number[] = [];
    for (const c of dbState.assignments) {
      if (c >= 0) {
        sizes[c] = (sizes[c] || 0) + 1;
      }
    }
    return sizes;
  }, [dbState]);

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

      {/* 2. PROGRESS / LEGEND */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        <div className="viz-ctrl__header"><span className="viz-ctrl__title">DBSCAN PROGRESS</span></div>
        <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)' }}>Algorithm Phase:</span>
              <span style={{ fontSize: '12px', fontWeight: 'bold', color: dbState?.phase === 'DONE' ? '#4ade80' : 'var(--c-primary)' }}>{dbState ? dbState.phase : 'READY'}</span>
            </div>
            {dbState && dbState.phase !== 'DONE' && (
              <div style={{ height: '4px', background: 'var(--c-surface-variant)', borderRadius: '2px', overflow: 'hidden' }}>
                 <div style={{ height: '100%', width: `${(dbState.currentPoint / points.length) * 100}%`, background: 'var(--c-primary)', transition: 'width 0.1s' }} />
              </div>
            )}
            
            <div style={{ display: 'flex', gap: '16px', marginTop: '12px', paddingTop: '12px', borderTop: '1px solid var(--c-panel-border)' }}>
               <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--c-on-surface-variant)' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#a855f7' }} /> Core Point
               </div>
               <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--c-on-surface-variant)' }}>
                  <span style={{ width: '10px', height: '10px', borderRadius: '50%', border: '2px solid #a855f7' }} /> Border Point
               </div>
               <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--c-on-surface-variant)' }}>
                  <span style={{ width: '10px', height: '10px', position: 'relative' }}>
                    <span style={{ position: 'absolute', top: '4px', left: 0, width: '10px', height: '2px', background: '#4b5563', transform: 'rotate(45deg)' }} />
                    <span style={{ position: 'absolute', top: '4px', left: 0, width: '10px', height: '2px', background: '#4b5563', transform: 'rotate(-45deg)' }} />
                  </span> Noise Point
               </div>
            </div>
        </div>
      </div>

      {/* 3. CLUSTER INFO */}
      {dbState && dbState.phase === 'DONE' && clusterSizes.length > 0 && (
        <div className="viz-scroll__section viz-scroll__section--controls">
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">CLUSTERS FOUND ({clusterSizes.length})</span></div>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            {clusterSizes.map((size, i) => size !== undefined && (
              <div key={i} style={{ padding: '8px 12px', borderRadius: '6px', background: `${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}18`, border: `1px solid ${CLUSTER_COLORS[i % CLUSTER_COLORS.length]}40`, textAlign: 'center', minWidth: '60px' }}>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: CLUSTER_COLORS[i % CLUSTER_COLORS.length] }}>{size}</div>
                <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)' }}>C{i + 1}</div>
              </div>
            ))}
            {dbState.noiseCount > 0 && (
              <div style={{ padding: '8px 12px', borderRadius: '6px', background: `#4b556318`, border: `1px solid #4b556340`, textAlign: 'center', minWidth: '60px' }}>
                <div style={{ fontSize: '14px', fontWeight: 'bold', color: '#9ca3af' }}>{dbState.noiseCount}</div>
                <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)' }}>Noise</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
