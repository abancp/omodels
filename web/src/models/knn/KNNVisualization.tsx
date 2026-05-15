import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateKNNData, classifyKNN, computeKNNMetrics, computeKNNConfusionMatrix, computeDataStats, type Point } from './math';

export default function KNNVisualization({
  params, dataset, datasetParams, resetVersion, onMetricsUpdate,
}: VisualizationProps) {
  const dataCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // State
  const [points, setPoints] = useState<Point[]>([]);

  // Viewport
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);

  // Inference
  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = useState<{x: number, y: number, prob: number, cls: number}[]>([]);

  // Params
  const k = (params.k as number) ?? 5;
  const metric = (params.metric as string) ?? 'euclidean';
  const weightType = (params.weights as string) ?? 'uniform';
  const showBoundaries = (params.showBoundaries as boolean) ?? true;
  const showNeighbors = (params.showNeighbors as boolean) ?? false;
  const pMinkowski = (params.p as number) ?? 2;
  const boundaryRes = (params.boundaryRes as number) ?? 6;
  const showVoronoi = (params.showVoronoi as boolean) ?? false;
  
  const numPoints = (datasetParams.points as number) ?? 60;
  const noise = (datasetParams.noise as number) ?? 0.2;

  const pushMetrics = useCallback((pts: Point[]) => {
    onMetricsUpdate(computeKNNMetrics(pts, k, metric, weightType, pMinkowski));
  }, [k, metric, weightType, pMinkowski, onMetricsUpdate]);

  /* Generate dataset */
  useEffect(() => {
    if (dataset === 'custom') {
      setPoints([]);
      return;
    }
    const pts = generateKNNData(dataset, numPoints, noise);
    setPoints(pts);
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1);
  }, [dataset, numPoints, noise]);

  /* Full reset */
  useEffect(() => {
    if (resetVersion === 0) return;
    setPoints([]);
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1);
    setHoverPt(null);
  }, [resetVersion]);

  /* Push metrics when params or points change */
  useEffect(() => { pushMetrics(points); }, [pushMetrics, points]);

  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current) return;
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const vp = vpRef.current;
    const px = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    
    const isClass1 = e.shiftKey || e.button === 2;
    const newPt = { x: px, y: py, cls: isClass1 ? 1 : 0 };
    setPoints(prev => [...prev, newPt]);
  }, []);

  /* Viewport handlers */
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const vp = vpRef.current;
    const mx = vp.xMin + (cx / rect.width) * (vp.xMax - vp.xMin);
    const my = vp.yMax - (cy / rect.height) * (vp.yMax - vp.yMin);
    vpRef.current = {
      xMin: mx + (vp.xMin - mx) * factor, xMax: mx + (vp.xMax - mx) * factor,
      yMin: my + (vp.yMin - my) * factor, yMax: my + (vp.yMax - my) * factor,
    };
    setVpVer(v => v + 1);
  }, []);

  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dataset === 'custom') return;
    dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } };
  }, [dataset]);

  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const vp = vpRef.current;
    const px = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    setHoverPt({ x: px, y: py, px: e.clientX - rect.left, py: e.clientY - rect.top });

    if (!dragRef.current) return;
    const dr = dragRef.current;
    const dx = ((e.clientX - dr.sx) / rect.width) * (dr.vp.xMax - dr.vp.xMin);
    const dy = ((e.clientY - dr.sy) / rect.height) * (dr.vp.yMax - dr.vp.yMin);
    vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy };
    setVpVer(v => v + 1);
  }, []);

  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);

  const resetView = useCallback(() => {
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1);
  }, []);

  const zoomBtn = useCallback((factor: number) => {
    const vp = vpRef.current;
    const mx = (vp.xMin + vp.xMax) / 2;
    const my = (vp.yMin + vp.yMax) / 2;
    vpRef.current = {
      xMin: mx + (vp.xMin - mx) * factor, xMax: mx + (vp.xMax - mx) * factor,
      yMin: my + (vp.yMin - my) * factor, yMax: my + (vp.yMax - my) * factor,
    };
    setVpVer(v => v + 1);
  }, []);

  /* Draw Data Canvas */
  useEffect(() => {
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId = 0;
    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      const W = rect.width;
      const H = rect.height;

      const vp = vpRef.current;
      const mapX = (x: number) => ((x - vp.xMin) / (vp.xMax - vp.xMin)) * W;
      const mapY = (y: number) => H - ((y - vp.yMin) / (vp.yMax - vp.yMin)) * H;

      ctx.clearRect(0, 0, W, H);

      const style = getComputedStyle(document.body);
      const gridColor = style.getPropertyValue('--c-grid').trim() || '#333';
      const textColor = style.getPropertyValue('--c-on-surface-variant').trim() || '#888';
      const primary = style.getPropertyValue('--c-primary').trim() || '#a855f7'; // Class 0
      const tertiary = style.getPropertyValue('--c-tertiary').trim() || '#e7c365'; // Class 1
      
      // Draw decision background
      if (showBoundaries && points.length > 0) {
        const res = Math.max(2, boundaryRes * 2); 
        for (let px = 0; px < W; px += res) {
          for (let py = 0; py < H; py += res) {
            const nx = vp.xMin + ((px + res/2) / W) * (vp.xMax - vp.xMin);
            const ny = vp.yMax - ((py + res/2) / H) * (vp.yMax - vp.yMin);
            const { cls, prob } = classifyKNN(nx, ny, points, k, metric, weightType, pMinkowski);
            
            ctx.fillStyle = cls === 1 ? `${tertiary}1A` : `${primary}1A`;
            if (weightType === 'distance' && prob > 0.5 && prob < 1) {
               ctx.globalAlpha = 0.5 + Math.abs(prob - 0.5);
            }
            ctx.fillRect(px, py, res, res);
            ctx.globalAlpha = 1.0;
          }
        }
      }

      // Draw Voronoi edges approximation via dense grid sampling 
      // (Only practical for 2D visual at coarse resolution)
      if (showVoronoi && points.length > 0) {
        ctx.beginPath();
        const vRes = 4;
        for (let px = 0; px < W; px += vRes) {
          for (let py = 0; py < H; py += vRes) {
            const nx = vp.xMin + (px / W) * (vp.xMax - vp.xMin);
            const ny = vp.yMax - (py / H) * (vp.yMax - vp.yMin);
            const ptClass = classifyKNN(nx, ny, points, 1, metric, 'uniform', pMinkowski).neighbors[0].point;
            const ptRight = classifyKNN(nx + vRes/W*(vp.xMax-vp.xMin), ny, points, 1, metric, 'uniform', pMinkowski).neighbors[0].point;
            const ptDown = classifyKNN(nx, ny - vRes/H*(vp.yMax-vp.yMin), points, 1, metric, 'uniform', pMinkowski).neighbors[0].point;
            
            if (ptClass !== ptRight || ptClass !== ptDown) {
              ctx.fillStyle = '#ffffff30';
              ctx.fillRect(px, py, vRes, vRes);
            }
          }
        }
      }

      // Grid lines
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      const getTicks = (min: number, max: number) => {
          const range = max - min;
          if (range <= 0) return [min];
          const step = Math.pow(10, Math.floor(Math.log10(range / 5)));
          const ticks = [];
          for (let t = Math.ceil(min / step) * step; t <= max; t += step) ticks.push(t);
          return ticks;
      };
      const xTicks = getTicks(vp.xMin, vp.xMax);
      for (const t of xTicks) {
        const px = mapX(t);
        ctx.moveTo(px, 0); ctx.lineTo(px, H);
      }
      const yTicks = getTicks(vp.yMin, vp.yMax);
      for (const t of yTicks) {
        const py = mapY(t);
        ctx.moveTo(0, py); ctx.lineTo(W, py);
      }
      ctx.stroke();

      ctx.fillStyle = textColor;
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textBaseline = 'top';
      for (const t of xTicks) ctx.fillText(t.toFixed(1), mapX(t) + 4, H - 16);
      ctx.textBaseline = 'bottom';
      for (const t of yTicks) ctx.fillText(t.toFixed(1), 4, mapY(t) - 4);

      // Lines to neighbors on hover
      if (showNeighbors && hoverPt && points.length > 0) {
        const { neighbors } = classifyKNN(hoverPt.x, hoverPt.y, points, k, metric, weightType, pMinkowski);
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;
        for (const n of neighbors) {
          ctx.beginPath();
          ctx.moveTo(mapX(hoverPt.x), mapY(hoverPt.y));
          ctx.lineTo(mapX(n.point.x), mapY(n.point.y));
          ctx.strokeStyle = n.point.cls === 0 ? `${primary}80` : `${tertiary}80`;
          ctx.stroke();
        }
        ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.arc(mapX(hoverPt.x), mapY(hoverPt.y), 6, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.stroke();
      }

      // Active Points
      for (const p of points) {
        ctx.beginPath();
        ctx.arc(mapX(p.x), mapY(p.y), 5, 0, Math.PI * 2);
        ctx.fillStyle = p.cls === 0 ? primary : tertiary;
        ctx.fill();
        ctx.strokeStyle = '#00000060';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Inference markers
      for (const ir of inferResults) {
        const sx = mapX(ir.x), sy = mapY(ir.y);
        ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fillStyle = ir.cls === 0 ? primary : tertiary; 
        ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2);
        ctx.strokeStyle = ir.cls === 0 ? `${primary}80` : `${tertiary}80`; 
        ctx.lineWidth = 1.5; ctx.stroke();
      }
    };
    
    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [points, vpVer, showBoundaries, showVoronoi, showNeighbors, hoverPt, inferResults, boundaryRes, k, metric, weightType, pMinkowski, dataset]);

  const handleInfer = useCallback(() => {
    const x = parseFloat(inferX);
    const y = parseFloat(inferY);
    if (isNaN(x) || isNaN(y)) return;
    const { cls, prob } = classifyKNN(x, y, points, k, metric, weightType, pMinkowski);
    setInferResults(prev => [{x, y, prob, cls}, ...prev].slice(0, 5));
  }, [inferX, inferY, points, k, metric, weightType, pMinkowski]);

  const stats = useMemo(() => computeDataStats(points), [points]);
  const cm = useMemo(() => computeKNNConfusionMatrix(points, k, metric, weightType, pMinkowski), [points, k, metric, weightType, pMinkowski]);

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER PLOT */}
      <div className="viz-scroll__section viz-scroll__section--canvas" style={{ position: 'relative' }}>
        <canvas
          ref={dataCanvasRef}
          className={`viz-canvas ${dataset !== 'custom' ? 'viz-canvas--pan' : 'viz-canvas--draw'}`}
          onContextMenu={e => e.preventDefault()}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={dataset === 'custom' ? handleDataClick : undefined}
          onWheel={handleWheel}
          style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }}
        />
        <div className="canvas__top-overlay">
           <div className="canvas__info-chip">
             <div className="canvas__dot"></div>
             <span className="canvas__info-model">KNN (K={k})</span>
             <span className="canvas__info-sep">|</span>
             <span className="canvas__info-detail">{metric}</span>
           </div>
        </div>
        {hoverPt && (
          <div className="viz-tooltip" style={{ left: hoverPt.px + 10, top: hoverPt.py - 24 }}>
            {hoverPt.x.toFixed(2)}, {hoverPt.y.toFixed(2)}
          </div>
        )}
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

      {/* 2. CONFUSION MATRIX */}
      <div className="viz-scroll__section viz-scroll__section--controls" style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">EVALUATION MATRIX</span>
            <span className="viz-ctrl__subtitle">Leave-one-out cross validation</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: '4px', textAlign: 'center', fontSize: '11px', maxWidth: '300px' }}>
            <div></div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 0</div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 1</div>
            <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 0</div>
            <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '14px' }}>{cm.tn}</div><div>TN</div>
            </div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{cm.fp}</div><div>FP</div>
            </div>
            <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 1</div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{cm.fn}</div><div>FN</div>
            </div>
            <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '14px' }}>{cm.tp}</div><div>TP</div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">INFERENCE</span>
          <span className="viz-ctrl__subtitle">Predict class using neighbors</span>
        </div>
        <div className="viz-infer__input-row">
          <div className="viz-infer__field">
            <label>Input x₁</label>
            <input className="viz-infer__input" type="number" step="0.01" value={inferX} onChange={e => setInferX(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} />
          </div>
          <div className="viz-infer__field">
            <label>Input x₂</label>
            <input className="viz-infer__input" type="number" step="0.01" value={inferY} onChange={e => setInferY(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} />
          </div>
          <button className="viz-infer__btn" onClick={handleInfer}>Predict</button>
          <div className="viz-infer__result" style={{ marginLeft: '10px' }}>
            <label>Prob</label>
            <span className="viz-infer__y">{inferResults.length > 0 ? inferResults[0].prob.toFixed(4) : '—'}</span>
          </div>
          <div className="viz-infer__result">
            <label>Class</label>
            <span className="viz-infer__y" style={{ color: inferResults.length > 0 ? (inferResults[0].cls === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)') : 'inherit' }}>
              {inferResults.length > 0 ? inferResults[0].cls : '—'}
            </span>
          </div>
        </div>
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr' }}>
              <span>x₁</span><span>x₂</span><span>Prob</span><span>Class</span>
            </div>
            {inferResults.map((r, i) => (
              <div key={i} className="viz-infer__history-row" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr' }}>
                <span>{r.x.toFixed(3)}</span><span>{r.y.toFixed(3)}</span><span>{r.prob.toFixed(4)}</span>
                <span style={{ color: r.cls === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)' }}>{r.cls}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 4. DATA STATISTICS */}
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">DATA STATISTICS</span>
          </div>
          <div className="viz-stats__grid">
            <div className="viz-stats__item"><span className="viz-stats__label">Total Points</span><span className="viz-stats__val">{stats.n}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 0</span><span className="viz-stats__val viz-stats__val--primary">{stats.nClass0}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 1</span><span className="viz-stats__val" style={{ color: 'var(--c-tertiary)' }}>{stats.nClass1}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">x₁ Range</span><span className="viz-stats__val">[{stats.xRange[0].toFixed(2)}, {stats.xRange[1].toFixed(2)}]</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">x₂ Range</span><span className="viz-stats__val">[{stats.yRange[0].toFixed(2)}, {stats.yRange[1].toFixed(2)}]</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
