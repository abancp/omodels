import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateSVMData, computeLoss, trainStep, computeMetrics, predict, formatEquation, computeConfusionMatrix, computeDataStats, computeMargin, initWeights, type Point, type Weights } from './math';

export default function SVMVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const scatterRef = useRef<HTMLDivElement>(null);

  // State
  const [points, setPoints] = useState<Point[]>([]);
  const degree = parseInt((params.degree as string) ?? '1', 10);
  
  const [weights, setWeights] = useState<Weights>([]);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [epochTarget, setEpochTarget] = useState(0);
  const [trained, setTrained] = useState(false);

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
  const learningRate = (params.learningRate as number) ?? 0.1;
  const epochs = (params.epochs as number) ?? 500;
  const cParam = (params.cParam as number) ?? 1.0;
  const numPoints = (datasetParams.points as number) ?? 100;
  const noise = (datasetParams.noise as number) ?? 0.15;

  const pushMetrics = useCallback((pts: Point[], w: Weights) => {
    onMetricsUpdate(computeMetrics(pts, w, degree));
  }, [degree, onMetricsUpdate]);

  const stepRef = useRef(0);
  const weightsRef = useRef<Weights>([]);
  const lossRef = useRef<number[]>([]);

  /* Generate dataset & reset model */
  useEffect(() => {
    if (dataset === 'custom') return;
    const pts = generateSVMData(dataset, numPoints, noise);
    setPoints(pts);
    const w = initWeights(degree);
    setWeights(w);
    weightsRef.current = w;
    const initLoss = [computeLoss(pts, w, degree, cParam)];
    setLossHistory(initLoss);
    lossRef.current = initLoss;
    setEpochTarget(0);
    setTrained(false);
    setInferResults([]);
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1);
    stepRef.current = 0;
  }, [dataset, numPoints, noise, degree, cParam]);

  /* Full reset */
  useEffect(() => {
    if (resetVersion === 0) return;
    const w = initWeights(degree, Date.now() % 10000);
    setWeights(w);
    weightsRef.current = w;
    const initLoss = [computeLoss(points, w, degree, cParam)];
    setLossHistory(initLoss);
    lossRef.current = initLoss;
    setEpochTarget(0);
    setTrained(false);
    setInferResults([]);
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1);
    setHoverPt(null);
    stepRef.current = 0;
  }, [resetVersion, points, degree, cParam]);

  /* Training Loop (PEGASOS SGD) — uses refs to avoid stale closures */
  useEffect(() => {
    if (!isTraining || points.length === 0) return;

    // Initialize from current ref (avoids stale closure)
    let w = trained ? [...weightsRef.current] : initWeights(degree, Date.now() % 10000);
    const prevLoss = [...lossRef.current];
    const target = prevLoss.length + epochs;
    setEpochTarget(target);
    stepRef.current = 0;

    let animId = 0;

    const loop = () => {
      const stepsPerFrame = Math.max(1, Math.floor(epochs / 120));
      for (let i = 0; i < stepsPerFrame && stepRef.current < epochs; i++) {
        const result = trainStep(w, points, degree, cParam, learningRate, stepRef.current);
        w = result.weights;
        prevLoss.push(result.loss);
        stepRef.current++;
      }

      weightsRef.current = w;
      lossRef.current = [...prevLoss];
      setWeights([...w]);
      setLossHistory([...prevLoss]);
      pushMetrics(points, w);

      if (stepRef.current < epochs) {
        animId = requestAnimationFrame(loop);
      } else {
        setTrained(true);
        onTrainingComplete();
      }
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isTraining]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current) return;
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const vp = vpRef.current;
    const px = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    
    const isClass1 = e.shiftKey || e.button === 2;
    setPoints(prev => [...prev, { x: px, y: py, cls: isClass1 ? 1 : 0 }]);
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
    if (!e.ctrlKey) return;
    dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } };
  }, []);

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

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
      const W = rect.width, H = rect.height;
      const padL = 44, padR = 12, padT = 12, padB = 28;
      const dw = W - padL - padR, dh = H - padT - padB;

      const vp = vpRef.current;
      const mapX = (x: number) => padL + ((x - vp.xMin) / (vp.xMax - vp.xMin)) * dw;
      const mapY = (y: number) => padT + ((vp.yMax - y) / (vp.yMax - vp.yMin)) * dh;

      ctx.clearRect(0, 0, W, H);

      const root = getComputedStyle(document.documentElement);
      const primary = root.getPropertyValue('--c-primary').trim() || '#cfbcff';
      const tertiary = root.getPropertyValue('--c-tertiary').trim() || '#e7c365';
      const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
      const border = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';

      // Nice tick steps
      const niceStep = (range: number) => {
        const raw = range / 5;
        const mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const n = raw / mag;
        return (n <= 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
      };
      const xStep = niceStep(vp.xMax - vp.xMin);
      const yStep = niceStep(vp.yMax - vp.yMin);

      // Grid + tick labels
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = 'center';
      const xStart = Math.ceil(vp.xMin / xStep) * xStep;
      for (let v = xStart; v <= vp.xMax; v += xStep) {
        const px = mapX(v);
        if (px < padL || px > W - padR) continue;
        ctx.strokeStyle = border; ctx.lineWidth = 0.4;
        ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, H - padB); ctx.stroke();
        ctx.fillStyle = muted; ctx.globalAlpha = 0.6;
        ctx.fillText(v.toFixed(xStep < 0.1 ? 2 : 1), px, H - padB + 14);
        ctx.globalAlpha = 1;
      }
      ctx.textAlign = 'right';
      const yStart = Math.ceil(vp.yMin / yStep) * yStep;
      for (let v = yStart; v <= vp.yMax; v += yStep) {
        const py = mapY(v);
        if (py < padT || py > H - padB) continue;
        ctx.strokeStyle = border; ctx.lineWidth = 0.4;
        ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(W - padR, py); ctx.stroke();
        ctx.fillStyle = muted; ctx.globalAlpha = 0.6;
        ctx.fillText(v.toFixed(yStep < 0.1 ? 2 : 1), padL - 4, py + 3);
        ctx.globalAlpha = 1;
      }

      // Axis lines
      ctx.strokeStyle = `${muted}40`; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB); ctx.stroke();

      // Clip data area
      ctx.save();
      ctx.beginPath(); ctx.rect(padL, padT, dw, dh); ctx.clip();

      // Margin heatmap
      if (weights.length > 0) {
        const res = 5;
        for (let px = padL; px < W - padR; px += res) {
          for (let py = padT; py < H - padB; py += res) {
            const nx = vp.xMin + ((px - padL + res / 2) / dw) * (vp.xMax - vp.xMin);
            const ny = vp.yMax - ((py - padT + res / 2) / dh) * (vp.yMax - vp.yMin);
            const md = computeMargin(nx, ny, weights, degree);
            if (md >= 1) ctx.fillStyle = `${tertiary}1A`;
            else if (md > 0) ctx.fillStyle = `${tertiary}33`;
            else if (md <= -1) ctx.fillStyle = `${primary}1A`;
            else ctx.fillStyle = `${primary}33`;
            ctx.fillRect(px, py, res, res);
          }
        }
        // Decision boundary contour (f(x)=0)
        ctx.strokeStyle = '#ffffff60'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        ctx.beginPath();
        let started = false;
        for (let px = padL; px <= W - padR; px += 2) {
          const nx = vp.xMin + ((px - padL) / dw) * (vp.xMax - vp.xMin);
          // Binary search for y where margin == 0
          let lo = vp.yMin, hi = vp.yMax, found = false;
          for (let iter = 0; iter < 30; iter++) {
            const mid = (lo + hi) / 2;
            const m = computeMargin(nx, mid, weights, degree);
            if (Math.abs(m) < 0.01) { found = true; lo = mid; break; }
            if (m > 0) hi = mid; else lo = mid;
          }
          if (found) {
            const py = mapY(lo);
            if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
          }
        }
        ctx.stroke(); ctx.setLineDash([]);
      }

      // Data points
      for (const p of points) {
        const sx = mapX(p.x), sy = mapY(p.y);
        ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = p.cls === 0 ? primary : tertiary;
        ctx.globalAlpha = 0.85; ctx.fill(); ctx.globalAlpha = 1;
        // Outer ring
        ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.strokeStyle = p.cls === 0 ? `${primary}40` : `${tertiary}40`;
        ctx.lineWidth = 0.5; ctx.stroke();

        // Support vector highlight
        const yTrue = p.cls === 1 ? 1 : -1;
        const md = computeMargin(p.x, p.y, weights, degree);
        if (1 - yTrue * md >= 0) {
          ctx.beginPath(); ctx.arc(sx, sy, 8, 0, Math.PI * 2);
          ctx.strokeStyle = '#ffffffaa'; ctx.lineWidth = 1.5; ctx.stroke();
        }
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
        // Crosshair lines
        ctx.setLineDash([2, 3]); ctx.strokeStyle = `${ir.cls === 0 ? primary : tertiary}40`; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(sx, H - padB); ctx.lineTo(sx, sy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(sx, sy); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore(); // end clip

      if (dataset === 'custom' && points.length < 3) {
        ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif";
        ctx.globalAlpha = 0.5; ctx.textAlign = 'center';
        ctx.fillText('Click to add points (Shift+click for class 1)', W / 2, H / 2);
        ctx.globalAlpha = 1; ctx.textAlign = 'start';
      }
    };
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [points, weights, vpVer, hoverPt, inferResults, degree, dataset]);

  /* Draw Loss Curve */
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
      const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
      const border = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = muted; ctx.font = "600 10px 'Inter', sans-serif";
      ctx.globalAlpha = 0.6; ctx.fillText('HINGE LOSS', 12, 18); ctx.globalAlpha = 1;

      const totalEp = epochTarget > 0 ? epochTarget : epochs;
      const padL = 40, padR = 16, padT = 30, padB = 24;
      const cw = w - padL - padR, ch = h - padT - padB;

      ctx.strokeStyle = border; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

      // X-axis ticks (always visible)
      ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
      for (let i = 0; i <= 4; i++) {
        const ep = Math.round((totalEp / 4) * i);
        const x = padL + (i / 4) * cw;
        ctx.fillText(String(ep), x - 6, h - 8);
      }

      if (lossHistory.length < 2) {
        ctx.globalAlpha = 0.3; ctx.font = "11px 'Inter', sans-serif"; ctx.textAlign = 'center';
        ctx.fillText('Train to see loss curve', w / 2, h / 2 + 10);
        ctx.globalAlpha = 1; ctx.textAlign = 'start';
        return;
      }

      const maxL = Math.max(...lossHistory) * 1.1;
      const minL = Math.max(0, Math.min(...lossHistory) * 0.9);

      // Y-axis ticks
      ctx.fillStyle = muted; ctx.globalAlpha = 0.5;
      for (let i = 0; i <= 4; i++) {
        const val = minL + (maxL - minL) * (1 - i / 4);
        const y = padT + (i / 4) * ch;
        ctx.fillText(val.toFixed(2), 2, y + 3);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.strokeStyle = border; ctx.lineWidth = 0.3; ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Loss line — x mapped to fixed totalEp range
      ctx.beginPath();
      for (let i = 0; i < lossHistory.length; i++) {
        const x = padL + (i / (totalEp - 1)) * cw;
        const y = padT + (1 - (lossHistory[i] - minL) / (maxL - minL)) * ch;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = '#e7c365'; ctx.lineWidth = 1.5; ctx.stroke();

      // Fill under
      const lastX = padL + ((lossHistory.length - 1) / (totalEp - 1)) * cw;
      const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
      gradient.addColorStop(0, '#e7c36520'); gradient.addColorStop(1, '#e7c36502');
      ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath();
      ctx.fillStyle = gradient; ctx.fill();

      // Progress indicator
      if (lossHistory.length < totalEp) {
        ctx.setLineDash([3, 3]); ctx.strokeStyle = '#e7c36550'; ctx.lineWidth = 0.8;
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
    const y = parseFloat(inferY);
    if (isNaN(x) || isNaN(y)) return;
    const { cls, prob } = predict(x, y, weights, degree);
    setInferResults(prev => [{x, y, prob, cls}, ...prev].slice(0, 5));
  }, [inferX, inferY, weights, degree]);

  const handleWeightChange = useCallback((index: number, val: string) => {
    const v = parseFloat(val);
    if (isNaN(v)) return;
    setWeights(prev => {
      const nw = [...prev];
      nw[index] = v;
      weightsRef.current = nw;
      pushMetrics(points, nw);
      return nw;
    });
  }, [points, pushMetrics]);

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen();
    else scatterRef.current?.requestFullscreen();
  }, []);

  const stats = useMemo(() => computeDataStats(points, weights, degree), [points, weights, degree]);
  const cm = useMemo(() => computeConfusionMatrix(points, weights, degree), [points, weights, degree]);
  const equationStr = useMemo(() => formatEquation(weights, degree), [weights, degree]);

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER PLOT */}
      <div className="viz-scroll__section viz-scroll__section--canvas" ref={scatterRef} style={{ position: 'relative' }}>
        <canvas
          ref={dataCanvasRef}
          className={`viz-canvas viz-canvas--draw`}
          onContextMenu={e => e.preventDefault()}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onClick={handleDataClick}
          onWheel={handleWheel}
          style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair' }}
        />
        <div className="canvas__top-overlay">
           <div className="canvas__info-chip">
             <div className="canvas__dot"></div>
             <span className="canvas__info-model">Soft-Margin SVM (Degree {degree})</span>
             <span className="canvas__info-sep">|</span>
             <span className="canvas__info-detail">C={cParam.toFixed(1)}</span>
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
          <button className="viz-scatter-btn" onClick={toggleFullscreen} title="Fullscreen">⛶</button>
        </div>
      </div>

      {/* 2. LOSS CURVE */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={lossCanvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. COEFFICIENT CONTROLS */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">BOUNDARY CONTROL</span>
            <span className="viz-ctrl__subtitle">Adjust weights manually</span>
          </div>
          <div className="viz-ctrl__equation-wrap">
            <div className="viz-ctrl__equation" title="Decision function">{equationStr}</div>
          </div>
          <div className="viz-ctrl__sliders">
            {weights.map((w, i) => (
              <div className="viz-ctrl__slider-row" key={i}>
                <label className="viz-ctrl__label">
                  {i === 0 ? 'b (Bias)' : `w${i} ${degree === 2 && i > 2 ? (i === 3 ? '(x₁²)' : i === 4 ? '(x₂²)' : '(x₁x₂)') : ''}`}
                </label>
                <span className="viz-ctrl__slider-val">{w.toFixed(3)}</span>
              </div>
            ))}
            {weights.map((w, i) => (
              <div key={`s${i}`}>
                <input type="range" className="control__range" min={-10} max={10} step={0.01} value={w}
                  onChange={e => handleWeightChange(i, e.target.value)} />
                <div className="control__range-labels"><span>-10</span><span>10</span></div>
              </div>
            ))}
          </div>
        </div>

        {/* CONFUSION MATRIX */}
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">EVALUATION MATRIX</span>
            <span className="viz-ctrl__subtitle">Classification performance</span>
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

      {/* 4. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">INFERENCE</span>
          <span className="viz-ctrl__subtitle">Predict class using trained margins</span>
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
            <label>Conf (σ)</label>
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
              <span>x₁</span><span>x₂</span><span>Conf</span><span>Class</span>
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

      {/* 5. DATA STATISTICS */}
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">DATA STATISTICS</span>
          </div>
          <div className="viz-stats__grid">
            <div className="viz-stats__item"><span className="viz-stats__label">N</span><span className="viz-stats__val">{stats.n}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 0</span><span className="viz-stats__val viz-stats__val--primary">{stats.nClass0}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 1</span><span className="viz-stats__val" style={{ color: 'var(--c-tertiary)' }}>{stats.nClass1}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Support Vectors</span><span className="viz-stats__val" style={{ color: 'var(--c-tertiary)' }}>{stats.supportVectors}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">x₁ Range</span><span className="viz-stats__val">[{stats.xRange[0].toFixed(2)}, {stats.xRange[1].toFixed(2)}]</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">x₂ Range</span><span className="viz-stats__val">[{stats.yRange[0].toFixed(2)}, {stats.yRange[1].toFixed(2)}]</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
