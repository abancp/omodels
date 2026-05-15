import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generatePolyData, computeLoss, computeGradients, computeMetrics, predict, formatEquation, type Point, type Weights } from './math';

export default function PolynomialRegressionVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  
  // State
  const [points, setPoints] = useState<Point[]>([]);
  const [weights, setWeights] = useState<Weights>([0.5, 0.5, 0]); // Init with degree 2
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [epochTarget, setEpochTarget] = useState(0);

  // Viewport
  const vpRef = useRef({ xMin: -0.08, xMax: 1.08, yMin: -0.5, yMax: 4.0 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);

  // Inference
  const [inferX, setInferX] = useState('0.50');
  const [inferResults, setInferResults] = useState<{x: number, y: number}[]>([]);

  // Params
  const degree = (params.degree as number) ?? 2;
  const lr = (params.learningRate as number) ?? 0.005;
  const epochs = (params.epochs as number) ?? 150;
  const showResiduals = (params.showResiduals as boolean) ?? false;
  const showTrueCurve = (params.showTrueCurve as boolean) ?? false;
  const numPoints = (datasetParams.points as number) ?? 60;
  const noise = (datasetParams.noise as number) ?? 0.25;

  const pushMetrics = useCallback((w: Weights) => {
    onMetricsUpdate(computeMetrics(points, w));
  }, [points, onMetricsUpdate]);

  /* Generate dataset */
  useEffect(() => {
    if (dataset === 'custom') return;
    const pts = generatePolyData(dataset, numPoints, noise);
    setPoints(pts);
    setWeights(new Array(degree + 1).fill(0).map((_, i) => i === 0 ? 0.5 : (i === 1 ? 0.5 : 0)));
    setLossHistory([]); setEpochTarget(0);
  }, [dataset, numPoints, noise, degree]);

  /* Handle Degree Change (resize weights) */
  useEffect(() => {
    setWeights(prev => {
      if (prev.length === degree + 1) return prev;
      const next = new Array(degree + 1).fill(0);
      for (let i = 0; i < Math.min(prev.length, degree + 1); i++) next[i] = prev[i];
      if (prev.length === 0) next[0] = 0.5; // Ensure bias has a starting value if it was empty
      return next;
    });
  }, [degree]);

  /* Full reset (triggered by store resetVersion) */
  useEffect(() => {
    if (resetVersion === 0) return;
    setWeights(new Array(degree + 1).fill(0).map((_, i) => i === 0 ? 0.5 : (i === 1 ? 0.5 : 0)));
    setLossHistory([]); setEpochTarget(0);
    vpRef.current = { xMin: -0.08, xMax: 1.08, yMin: -0.5, yMax: 4.0 };
    setVpVer(v => v + 1);
    setHoverPt(null);
  }, [resetVersion, degree]);

  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current) return;
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const vp = vpRef.current;
    const px = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    setPoints(prev => [...prev, { x: px, y: py }]);
  }, []);

  /* Viewport handlers */
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
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
    vpRef.current = { xMin: -0.08, xMax: 1.08, yMin: -0.5, yMax: 4.0 };
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



  /* Training Loop */
  useEffect(() => {
    if (!isTraining) return;
    if (points.length < 2) { onTrainingComplete(); return; }

    const lossType = params.lossFunction as string ?? 'mse';
    const regularization = params.regularization as string ?? 'none';
    const regStrength = params.regStrength as number ?? 0.01;
    const gradClipping = params.gradClipping as boolean ?? true;

    let w = [...weights];
    let epoch = 0;
    let animId = 0;
    const prevLoss = [...lossHistory];
    const totalTarget = prevLoss.length + epochs;
    setEpochTarget(totalTarget);

    const step = () => {
      const stepsPerFrame = Math.max(1, Math.floor(epochs / 120));
      for (let s = 0; s < stepsPerFrame && epoch < epochs; s++, epoch++) {
        const grads = computeGradients(points, w, lossType);

        // Regularization
        if (regularization === 'l2') {
          for (let j = 1; j <= degree; j++) grads[j] += regStrength * w[j];
        } else if (regularization === 'l1') {
          for (let j = 1; j <= degree; j++) grads[j] += regStrength * Math.sign(w[j]);
        } // elastic net omitted for brevity in MVP

        // Gradient clipping
        if (gradClipping) {
          for (let j = 0; j <= degree; j++) {
            grads[j] = Math.max(-5.0, Math.min(5.0, grads[j]));
          }
        }

        // Update weights
        for (let j = 0; j <= degree; j++) {
          w[j] -= lr * grads[j];
        }

        prevLoss.push(computeLoss(points, w, lossType));
      }
      setWeights([...w]);
      setLossHistory([...prevLoss]);
      pushMetrics(w);

      if (epoch < epochs) { animId = requestAnimationFrame(step); }
      else { onTrainingComplete(); }
    };
    animId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animId);
  }, [isTraining]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { pushMetrics(weights); }, [points]); // eslint-disable-line

  /* Draw Data Canvas */
  useEffect(() => {
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high DPI
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

    // CSS vars
    const style = getComputedStyle(document.body);
    const gridColor = style.getPropertyValue('--c-grid').trim() || '#333';
    const textColor = style.getPropertyValue('--c-on-surface-variant').trim() || '#888';
    const pointColor = '#fcd34d';
    const pointStroke = '#b45309';
    const primary = style.getPropertyValue('--c-primary').trim() || '#a855f7';
    const residualColor = 'rgba(168, 85, 247, 0.4)';

    // Grid lines
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const xTicks = getTicks(vp.xMin, vp.xMax, 5);
    for (const t of xTicks) {
      const px = mapX(t);
      ctx.moveTo(px, 0); ctx.lineTo(px, H);
    }
    const yTicks = getTicks(vp.yMin, vp.yMax, 5);
    for (const t of yTicks) {
      const py = mapY(t);
      ctx.moveTo(0, py); ctx.lineTo(W, py);
    }
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = textColor;
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textBaseline = 'top';
    for (const t of xTicks) ctx.fillText(t.toFixed(1), mapX(t) + 4, H - 16);
    ctx.textBaseline = 'bottom';
    for (const t of yTicks) ctx.fillText(t.toFixed(1), 4, mapY(t) - 4);

    // Residuals
    if (showResiduals && points.length > 0) {
      ctx.beginPath();
      for (const p of points) {
        ctx.moveTo(mapX(p.x), mapY(p.y));
        ctx.lineTo(mapX(p.x), mapY(predict(p.x, weights)));
      }
      ctx.strokeStyle = residualColor;
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Polynomial Curve
    const curvePoints = 200;
    ctx.beginPath();
    for (let i = 0; i <= curvePoints; i++) {
      const x = vp.xMin + (i / curvePoints) * (vp.xMax - vp.xMin);
      const y = predict(x, weights);
      const px = mapX(x);
      const py = mapY(y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = primary;
    ctx.lineWidth = 2;
    ctx.stroke();

    // True curve (if requested)
    if (showTrueCurve && dataset !== 'custom') {
      ctx.beginPath();
      for (let i = 0; i <= curvePoints; i++) {
        const x = vp.xMin + (i / curvePoints) * (vp.xMax - vp.xMin);
        let y = 0;
        if (dataset === 'quadratic') y = 2 * x * x - x + 0.5;
        else if (dataset === 'cubic') y = 4 * x * x * x - 6 * x * x + 3 * x + 0.2;
        else if (dataset === 'sinusoidal') y = Math.sin(x * Math.PI * 2) * 1.5 + 2.0;
        else if (dataset === 'step') y = x > 0.5 ? 3.0 : 1.0;
        else if (dataset === 'noisy') y = 2 * x * x - x + 0.5;
        else y = 2 * x + 0.5; // linear default
        const px = mapX(x);
        const py = mapY(y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = '#22c55e'; // Green for ground truth
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Points
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(mapX(p.x), mapY(p.y), 4, 0, Math.PI * 2);
      ctx.fillStyle = pointColor;
      ctx.fill();
      ctx.strokeStyle = pointStroke;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Inference markers
    for (const ir of inferResults) {
      const sx = mapX(ir.x), sy = mapY(ir.y);
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ff6b6b'; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(sx, sy, 9, 0, Math.PI * 2);
      ctx.strokeStyle = '#ff6b6b60'; ctx.lineWidth = 1; ctx.stroke();
      ctx.setLineDash([2, 3]); ctx.strokeStyle = '#ff6b6b40'; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(sx, H - 4); ctx.lineTo(sx, sy); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(4, sy); ctx.lineTo(sx, sy); ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [points, weights, vpVer, showResiduals, showTrueCurve, dataset, inferResults]);

  /* Draw Loss Curve (fixed x-axis) */
  useEffect(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      const w = rect.width, h = rect.height;
      const root = getComputedStyle(document.documentElement);
      const primary = root.getPropertyValue('--c-primary').trim() || '#cfbcff';
      const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
      const border = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = muted; ctx.font = "600 10px 'Inter', sans-serif";
      ctx.globalAlpha = 0.6; ctx.fillText('LOSS CURVE', 12, 18); ctx.globalAlpha = 1;

      const totalEp = epochTarget > 0 ? epochTarget : epochs;

      if (lossHistory.length < 1) {
        // Draw empty axis frame with target epochs
        const padL = 40, padR = 16, padT = 30, padB = 24;
        ctx.strokeStyle = border; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
        // X-axis ticks
        ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
        for (let i = 0; i <= 4; i++) {
          const ep = Math.round((totalEp / 4) * i);
          const x = padL + (i / 4) * (w - padL - padR);
          ctx.fillText(String(ep), x - 6, h - 8);
        }
        ctx.globalAlpha = 1;
        ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
        ctx.fillText('Train to see loss curve', w / 2, h / 2 + 10);
        ctx.globalAlpha = 1; ctx.textAlign = 'start';
        return;
      }

      const padL = 40, padR = 16, padT = 30, padB = 24;
      const cw = w - padL - padR, ch = h - padT - padB;

      ctx.strokeStyle = border; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

      const maxLoss = Math.max(...lossHistory) * 1.1;

      // Y-axis ticks
      ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
      for (let i = 0; i <= 4; i++) {
        const val = maxLoss * (1 - i / 4);
        const y = padT + (i / 4) * ch;
        ctx.fillText(val.toFixed(2), 2, y + 3);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.strokeStyle = border; ctx.lineWidth = 0.3; ctx.stroke();
      }

      // X-axis ticks
      for (let i = 0; i <= 4; i++) {
        const ep = Math.round((totalEp / 4) * i);
        const x = padL + (i / 4) * cw;
        ctx.fillText(String(ep), x - 6, h - 8);
      }
      ctx.globalAlpha = 1;

      // Loss line
      ctx.beginPath();
      for (let i = 0; i < lossHistory.length; i++) {
        const x = padL + (i / (totalEp - 1)) * cw;
        const y = padT + ((maxLoss - lossHistory[i]) / maxLoss) * ch;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = primary; ctx.lineWidth = 1.5; ctx.stroke();

      // Fill under
      const lastX = padL + ((lossHistory.length - 1) / (totalEp - 1)) * cw;
      const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
      gradient.addColorStop(0, `${primary}20`); gradient.addColorStop(1, `${primary}02`);
      ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath();
      ctx.fillStyle = gradient; ctx.fill();

      // Progress indicator
      if (lossHistory.length < totalEp) {
        ctx.setLineDash([3, 3]); ctx.strokeStyle = `${primary}50`; ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.moveTo(lastX, padT); ctx.lineTo(lastX, h - padB); ctx.stroke();
        ctx.setLineDash([]);
      }
    };
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [lossHistory, epochTarget, epochs]);

  const handleInfer = useCallback(() => {
    const x = parseFloat(inferX);
    if (isNaN(x)) return;
    const y = predict(x, weights);
    setInferResults(prev => [{x, y}, ...prev].slice(0, 5));
  }, [inferX, weights]);

  // Compute stats
  const stats = useMemo(() => {
    if (points.length < 2) return null;
    const n = points.length;
    let sumX = 0, sumY = 0, xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of points) {
      sumX += p.x; sumY += p.y;
      if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
    }
    const mx = sumX / n; const my = sumY / n;
    let cov = 0, varX = 0, varY = 0;
    for (const p of points) {
      const dx = p.x - mx, dy = p.y - my;
      cov += dx * dy; varX += dx * dx; varY += dy * dy;
    }
    const r = (varX && varY) ? cov / Math.sqrt(varX * varY) : 0;
    return { n, mx, my, xMin, xMax, yMin, yMax, r };
  }, [points]);

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER PLOT */}
      <div className="viz-scroll__section viz-scroll__section--canvas">
        <canvas
          ref={dataCanvasRef}
          className={`viz-canvas ${dataset !== 'custom' ? 'viz-canvas--pan' : 'viz-canvas--draw'}`}
          onClick={dataset === 'custom' ? handleDataClick : undefined}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }}
        />
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

      {/* 2. LOSS CURVE */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={lossCanvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. DEGREE VISUALIZER */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">DEGREE VISUALIZER</span>
          <span className="viz-ctrl__subtitle">Compare polynomial complexities</span>
        </div>
        <div className="viz-degree-strip" style={{ display: 'flex', gap: '8px', overflowX: 'auto', padding: '4px 0' }}>
          {[1, 2, 3, 4, 5, 6, 7, 8].map(d => (
            <div 
              key={d}
              className={`viz-degree-thumb ${degree === d ? 'viz-degree-thumb--active' : ''}`}
              style={{ 
                width: '64px', height: '48px', borderRadius: '4px', position: 'relative', flexShrink: 0,
                border: `1px solid ${degree === d ? 'var(--c-primary)' : 'var(--c-panel-border)'}`,
                background: degree === d ? 'rgba(168, 85, 247, 0.1)' : 'var(--c-surface)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                boxShadow: degree === d ? '0 0 0 1px var(--c-primary)' : 'none'
              }}
            >
              <span style={{ fontSize: '14px', fontWeight: 600, color: degree === d ? 'var(--c-primary)' : 'var(--c-on-surface)' }}>x^{d}</span>
              <div style={{ position: 'absolute', bottom: '2px', width: '100%', textAlign: 'center', fontSize: '9px', color: 'var(--c-on-surface-variant)' }}>
                {d === 1 ? 'Line' : d === 2 ? 'Quad' : d === 3 ? 'Cubic' : `Deg ${d}`}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 4. COEFFICIENT CONTROL */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">COEFFICIENT CONTROL</span>
          <span className="viz-ctrl__subtitle">Adjust weights manually</span>
        </div>
        <div className="viz-ctrl__split">
          <div className="viz-ctrl__right" style={{ flex: 1 }}>
            <div className="viz-ctrl__equation-wrap">
              <div className="viz-ctrl__equation">
                {formatEquation(weights)}
              </div>
            </div>
            <div className="viz-ctrl__sliders">
              {weights.map((w, i) => (
                <div key={i} style={{ marginBottom: '8px' }}>
                  <div className="viz-ctrl__slider-row">
                    <label>{i === 0 ? 'Bias (w₀)' : `w${i} (x^${i})`}</label>
                    <span className="viz-ctrl__slider-val">{w.toFixed(3)}</span>
                  </div>
                  <input
                    type="range" className="control__range"
                    min="-5" max="5" step="0.01"
                    value={w}
                    onChange={(e) => {
                      const next = [...weights];
                      next[i] = parseFloat(e.target.value);
                      setWeights(next);
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 5. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">INFERENCE</span>
          <span className="viz-ctrl__subtitle">Predict y for a given x</span>
        </div>
        <div className="viz-infer__input-row">
          <div className="viz-infer__field">
            <label>Input x</label>
            <input className="viz-infer__input" type="number" step="0.01" value={inferX} onChange={e => setInferX(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} />
          </div>
          <button className="viz-infer__btn" onClick={handleInfer}>Predict</button>
          <div className="viz-infer__result">
            <label>ŷ</label>
            <span className="viz-infer__y">{inferResults.length > 0 ? inferResults[0].y.toFixed(4) : '—'}</span>
          </div>
        </div>
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header">
              <span>x</span><span>ŷ</span>
            </div>
            {inferResults.map((r, i) => (
              <div key={i} className="viz-infer__history-row">
                <span>{r.x.toFixed(3)}</span><span>{r.y.toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 6. DATA STATISTICS */}
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">DATA STATISTICS</span>
          </div>
          <div className="viz-stats__grid">
            <div className="viz-stats__item"><span className="viz-stats__label">N</span><span className="viz-stats__val">{stats.n}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Correlation</span><span className="viz-stats__val viz-stats__val--primary">{stats.r.toFixed(4)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean X</span><span className="viz-stats__val">{stats.mx.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean Y</span><span className="viz-stats__val">{stats.my.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">X Range</span><span className="viz-stats__val">[{stats.xMin.toFixed(2)}, {stats.xMax.toFixed(2)}]</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Y Range</span><span className="viz-stats__val">[{stats.yMin.toFixed(2)}, {stats.yMax.toFixed(2)}]</span></div>
          </div>
        </div>
      )}
    </div>
  );
}

// Utility for nice axis ticks
function getTicks(min: number, max: number, maxTicks = 5): number[] {
  const range = max - min;
  if (range <= 0) return [min];
  const stepMag = Math.pow(10, Math.floor(Math.log10(range / maxTicks)));
  const stepNorm = (range / maxTicks) / stepMag;
  let step = stepMag;
  if (stepNorm > 5) step *= 10;
  else if (stepNorm > 2) step *= 5;
  else step *= 2;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let t = start; t <= max; t += step) ticks.push(t);
  return ticks;
}
