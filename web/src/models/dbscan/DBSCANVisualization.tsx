import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClusteringData, type Point, initDBSCAN, dbscanStep, computeMetrics, type DBSCANState, getDist } from './math';
import { drawDataCanvas, CLUSTER_COLORS } from './drawHelpers';
import { usePlayground } from '../../store';
import { usePersistentState } from '../../hooks/usePersistentState';

export default function DBSCANVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = usePersistentState<Point[]>('omodels-dbscan-points', []);
  const [dbState, setDbState] = usePersistentState<DBSCANState | null>('omodels-dbscan-dbState', null);
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);
  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = usePersistentState<{x: number; y: number; cluster: number; dist: number}[]>('omodels-dbscan-inferResults', []);

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

  // Import from store
  const { importedData, importVersion, testData, testVersion, setTestResults } = usePlayground();

  // Test dataset evaluation for DBSCAN
  useEffect(() => {
    if (!testData || testData.length === 0 || !dbState) return;
    const total = testData.length;
    const results: Record<string, any> = { total, predictions: [] };

    let totalDist = 0;
    let coreAssignedCount = 0;

    for (const p of testData) {
      const x = p.x !== undefined ? p.x : (p.features?.[0] ?? 0.5);
      const y = p.y !== undefined ? p.y : (p.features?.[1] ?? 0.5);
      
      // Predict using closest core point within eps
      let bestDist = Infinity;
      let assignedCluster = -1;
      
      for (let i = 0; i < points.length; i++) {
        if (dbState.pointTypes[i] === 'core') {
          const d = getDist({ x, y }, points[i], metric);
          if (d <= eps && d < bestDist) {
            bestDist = d;
            assignedCluster = dbState.assignments[i];
          }
        }
      }
      
      if (assignedCluster >= 0) {
        totalDist += bestDist;
        coreAssignedCount++;
      }
      
      results.predictions.push({ features: [x, y], actual: assignedCluster, predicted: assignedCluster });
    }

    results.type = 'clustering';
    results.avgCentroidDist = coreAssignedCount > 0 ? totalDist / coreAssignedCount : 0;
    results.silhouetteScore = dbState.noiseCount / Math.max(1, points.length);
    setTestResults(results);
  }, [testVersion, testData, dbState, points, eps, metric]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (dataset !== 'import' || !importedData || importedData.length === 0) return;
    const pts = (importedData as any[]).map((p: any) => ({ x: p.x, y: p.y }));
    setPoints(pts);
    // Reset model state for clean start
    setDbState(null); setInferResults([]);
    // Auto-zoom viewport
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.15 || 0.5, yPad = (yMax - yMin) * 0.15 || 0.5;
    vpRef.current = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
    setVpVer(v => v + 1);
  }, [importVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (dataset === 'custom' || dataset === 'import') return; setPoints(generateClusteringData(dataset, numPoints, noise)); setDbState(null); setInferResults([]); }, [dataset, numPoints, noise]);
  useEffect(() => { if (resetVersion === 0) return; setDbState(null); setInferResults([]); vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, [resetVersion]);
  useEffect(() => { setDbState(null); }, [eps, minPts, metric]); // reset if params change
  
  // Mouse handlers
  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dragRef.current) return; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setPoints(prev => [...prev, { x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin) }]); setDbState(null); }, []);
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => { if (!e.ctrlKey) return; e.preventDefault(); const f = e.deltaY > 0 ? 1.1 : 0.9; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; const mx = vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin); const my = vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin); vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dataset === 'custom' || dataset === 'import') return; dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; }, [dataset]);
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
  useEffect(() => { const c = dataRef.current; if (!c) return; const r = () => drawDataCanvas(c, points, dbState, vpRef.current, hoverPt, dataset, showEpsCircles, showPointTypes, (testData || undefined) as Point[] | undefined); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, dbState, vpVer, hoverPt, dataset, showEpsCircles, showPointTypes, testData]);

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

  // Inference logic
  const handleInfer = useCallback(() => {
    if (!dbState) return;
    const x = parseFloat(inferX), y = parseFloat(inferY); if (isNaN(x) || isNaN(y)) return;
    
    // Find closest core point within eps
    let bestDist = Infinity;
    let assignedCluster = -1; // Default: noise (-1)
    
    for (let i = 0; i < points.length; i++) {
      if (dbState.pointTypes[i] === 'core') {
        const d = getDist({ x, y }, points[i], metric);
        if (d <= eps && d < bestDist) {
          bestDist = d;
          assignedCluster = dbState.assignments[i];
        }
      }
    }
    
    setInferResults(prev => [{ x, y, cluster: assignedCluster, dist: bestDist === Infinity ? 0 : bestDist }, ...prev].slice(0, 5));
  }, [inferX, inferY, dbState, points, eps, metric]); // eslint-disable-line react-hooks/exhaustive-deps

  // Neighborhood density telemetry on hover
  const hoverNeighbors = useMemo(() => {
    if (!hoverPt || points.length === 0) return null;
    const neighbors = [];
    for (let i = 0; i < points.length; i++) {
      const d = getDist(hoverPt, points[i], metric);
      if (d <= eps) {
        neighbors.push({ index: i, dist: d, type: dbState ? dbState.pointTypes[i] : 'unassigned' });
      }
    }
    return neighbors;
  }, [hoverPt, points, eps, metric, dbState]);

  const stats = useMemo(() => {
    if (points.length === 0) return null;
    const n = points.length;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const meanX = xs.reduce((a, b) => a + b, 0) / n;
    const meanY = ys.reduce((a, b) => a + b, 0) / n;
    const stdX = Math.sqrt(xs.map(x => (x - meanX) ** 2).reduce((a, b) => a + b, 0) / n);
    const stdY = Math.sqrt(ys.map(y => (y - meanY) ** 2).reduce((a, b) => a + b, 0) / n);
    return { n, meanX, meanY, stdX, stdY };
  }, [points]);

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
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">CLUSTERS FOUND ({clusterSizes.filter(s => s !== undefined).length})</span></div>
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

      {/* 4. DBSCAN ALGORITHM TRACKER */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">DBSCAN ALGORITHM TRACKER</span>
          <span className="viz-ctrl__subtitle">Mathematical rules & core density mechanics</span>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '12px' }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', color: 'var(--c-on-surface-variant)', background: 'rgba(0,0,0,0.2)', padding: '10px', overflowX: 'auto', whiteSpace: 'nowrap' }}>
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Point Core/Border/Noise Criteria</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Neighborhood: <span style={{ color: '#a855f7' }}>N_ε(p) = {"{ q ∈ D | dist(p, q) ≤ ε }"}</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Core Point: <span style={{ color: 'var(--c-primary)' }}>|N_ε(p)| ≥ MinPts ({minPts})</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Border Point: <span style={{ color: 'var(--c-tertiary)' }}>|N_ε(p)| &lt; MinPts and p ∈ N_ε(c) for Core c</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Noise Point: <span style={{ color: '#9ca3af' }}>Neither Core nor Border</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>2. Density Reachability & Connectivity</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Density-Reachable: <span style={{ color: '#4ade80' }}>p is reachable from q if path exists via Cores</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Density-Connected: <span style={{ color: '#38bdf8' }}>p & q are connected if both reachable from Core c</span></div>
          </div>

          {hoverPt && hoverNeighbors && (
            <div style={{ background: 'var(--c-surface-variant)', padding: '10px', border: '1px solid var(--c-panel-border)', borderRadius: '4px' }}>
              <div style={{ fontWeight: 'bold', fontSize: '12px', color: 'var(--c-primary)', marginBottom: '6px' }}>
                Neighborhood Density Telemetry at ({hoverPt.x.toFixed(3)}, {hoverPt.y.toFixed(3)})
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px' }}>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '6px', fontSize: '10px', borderLeft: `3px solid var(--c-tertiary)` }}>
                  <div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold' }}>Density (ε={eps})</div>
                  <div style={{ marginTop: '4px', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Neighbors:</span>
                    <span style={{ fontFamily: 'monospace', fontWeight: 'bold', color: 'var(--c-error)' }}>{hoverNeighbors.length} pts</span>
                  </div>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.2)', padding: '6px', fontSize: '10px', borderLeft: `3px solid ${hoverNeighbors.length >= minPts ? '#4ade80' : '#fb923c'}` }}>
                  <div style={{ color: hoverNeighbors.length >= minPts ? '#4ade80' : '#fb923c', fontWeight: 'bold' }}>Core-Eligible</div>
                  <div style={{ marginTop: '4px', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Required:</span>
                    <span style={{ fontFamily: 'monospace' }}>{minPts} pts</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 5. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header"><span className="viz-ctrl__title">INFERENCE</span><span className="viz-ctrl__subtitle">Predict density cluster assignment</span></div>
        <div className="viz-infer__input-row">
          <div className="viz-infer__field"><label>Input x₁</label><input className="viz-infer__input" type="number" step="0.01" value={inferX} onChange={e => setInferX(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <div className="viz-infer__field"><label>Input x₂</label><input className="viz-infer__input" type="number" step="0.01" value={inferY} onChange={e => setInferY(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <button className="viz-infer__btn" onClick={handleInfer} disabled={!dbState}>Predict</button>
          <div className="viz-infer__result" style={{ marginLeft: '10px' }}><label>Cluster</label><span className="viz-infer__y" style={{ color: inferResults.length > 0 ? (inferResults[0].cluster >= 0 ? CLUSTER_COLORS[inferResults[0].cluster % CLUSTER_COLORS.length] : 'var(--c-on-surface-variant)') : 'inherit', fontWeight: 'bold' }}>{inferResults.length > 0 ? (inferResults[0].cluster >= 0 ? `C${inferResults[0].cluster + 1}` : 'Noise') : '—'}</span></div>
          <div className="viz-infer__result"><label>Distance</label><span className="viz-infer__y">{inferResults.length > 0 ? (inferResults[0].cluster >= 0 ? inferResults[0].dist.toFixed(4) : '—') : '—'}</span></div>
        </div>
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr' }}><span>x₁</span><span>x₂</span><span>Cluster</span><span>Dist</span></div>
            {inferResults.map((r, i) => (
              <div key={i} className="viz-infer__history-row" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr' }}>
                <span>{r.x.toFixed(3)}</span><span>{r.y.toFixed(3)}</span>
                <span style={{ color: r.cluster >= 0 ? CLUSTER_COLORS[r.cluster % CLUSTER_COLORS.length] : '#9ca3af', fontWeight: 'bold' }}>{r.cluster >= 0 ? `C${r.cluster + 1}` : 'Noise'}</span>
                <span>{r.cluster >= 0 ? r.dist.toFixed(4) : '—'}</span>
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
