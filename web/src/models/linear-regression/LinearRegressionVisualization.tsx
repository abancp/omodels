import { useRef, useEffect, useState, useCallback, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps, MetricValue } from '../registry';

interface Point { x: number; y: number; }

function seededRandom(seed: number) {
  let s = seed;
  return () => { s = (s * 16807 + 0) % 2147483647; return s / 2147483647; };
}

function generateData(dataset: string, count: number, noise: number): Point[] {
  const rand = seededRandom(count * 100 + Math.floor(noise * 50));
  const pts: Point[] = [];
  for (let i = 0; i < count; i++) {
    const x = rand();
    let y: number;
    switch (dataset) {
      case 'noisy': y = 2 * x + 0.5 + (rand() - 0.5) * noise * 3; break;
      case 'outliers': {
        y = 2 * x + 0.5 + (rand() - 0.5) * noise * 0.8;
        if (rand() < 0.08) y += (rand() > 0.5 ? 1 : -1) * (1.5 + rand());
        break;
      }
      default: y = 2 * x + 0.5 + (rand() - 0.5) * noise * 1.2;
    }
    pts.push({ x, y: Math.max(0, Math.min(3.5, y)) });
  }
  return pts;
}

function computeMetrics(points: Point[], m: number, b: number): MetricValue[] {
  if (points.length < 2) return [
    { label: 'R²', value: '—', isPrimary: true },
    { label: 'MSE', value: '—' }, { label: 'MAE', value: '—' },
    { label: 'Equation', value: '—' },
  ];
  const yMean = points.reduce((s, p) => s + p.y, 0) / points.length;
  let ssTot = 0, ssRes = 0, mse = 0, mae = 0;
  for (const p of points) {
    const pred = m * p.x + b;
    ssRes += (p.y - pred) ** 2; ssTot += (p.y - yMean) ** 2;
    mse += (p.y - pred) ** 2; mae += Math.abs(p.y - pred);
  }
  mse /= points.length; mae /= points.length;
  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const sign = b >= 0 ? '+' : '−';
  return [
    { label: 'R²', value: r2.toFixed(4), isPrimary: true },
    { label: 'MSE', value: mse.toFixed(4) },
    { label: 'MAE', value: mae.toFixed(4) },
    { label: 'Equation', value: `y = ${m.toFixed(2)}x ${sign} ${Math.abs(b).toFixed(2)}` },
  ];
}

export default function LinearRegressionVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLCanvasElement>(null);
  const scatterRef = useRef<HTMLDivElement>(null);

  const [points, setPoints] = useState<Point[]>([]);
  const [weights, setWeights] = useState({ m: 0.5, b: 0.5 });
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [epochTarget, setEpochTarget] = useState(0);
  const [trained, setTrained] = useState(false);

  // Viewport: pan/zoom
  const vpRef = useRef({ xMin: -0.08, xMax: 1.08, yMin: -0.2, yMax: 3.7 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);

  // Hover tooltip
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);

  // Inference
  const [inferX, setInferX] = useState('0.50');
  const [inferResults, setInferResults] = useState<{ x: number; y: number }[]>([]);

  // Equation editing
  const [editingEq, setEditingEq] = useState(false);
  const [editM, setEditM] = useState('');
  const [editB, setEditB] = useState('');

  const lr = (params.learningRate as number) ?? 0.01;
  const epochs = (params.epochs as number) ?? 100;
  const showResiduals = (params.showResiduals as boolean) ?? false;
  const numPoints = (datasetParams.points as number) ?? 50;
  const noise = (datasetParams.noise as number) ?? 0.3;

  /* Generate dataset */
  useEffect(() => {
    if (dataset === 'custom') return;
    const pts = generateData(dataset, numPoints, noise);
    setPoints(pts);
    setWeights({ m: 0.5, b: 0.5 });
    setLossHistory([]); setEpochTarget(0);
    setTrained(false); setInferResults([]);
  }, [dataset, numPoints, noise]);

  /* Full reset (triggered by store resetVersion) */
  useEffect(() => {
    if (resetVersion === 0) return;
    setWeights({ m: 0.5, b: 0.5 });
    setLossHistory([]); setEpochTarget(0);
    setTrained(false); setInferResults([]);
    vpRef.current = { xMin: -0.08, xMax: 1.08, yMin: -0.2, yMax: 3.7 };
    setVpVer(v => v + 1);
    setHoverPt(null); setEditingEq(false);
  }, [resetVersion]);

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

  /* Viewport: wheel zoom */
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const vp = vpRef.current;
    const factor = e.deltaY > 0 ? 1.12 : 0.88;
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const my = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    vpRef.current = {
      xMin: mx + (vp.xMin - mx) * factor, xMax: mx + (vp.xMax - mx) * factor,
      yMin: my + (vp.yMin - my) * factor, yMax: my + (vp.yMax - my) * factor,
    };
    setVpVer(v => v + 1);
  }, []);

  /* Viewport: drag pan */
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dataset === 'custom') return;
    dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } };
  }, [dataset]);
  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    // Hover tooltip
    const vp = vpRef.current;
    const mx = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const my = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    let nearest: typeof hoverPt = null;
    let minD = 0.04;
    for (const p of points) {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < minD) { minD = d; nearest = { x: p.x, y: p.y, px: e.clientX - rect.left, py: e.clientY - rect.top }; }
    }
    setHoverPt(nearest);
    // Drag
    const dr = dragRef.current;
    if (!dr) return;
    const dx = ((e.clientX - dr.sx) / rect.width) * (dr.vp.xMax - dr.vp.xMin);
    const dy = ((e.clientY - dr.sy) / rect.height) * (dr.vp.yMax - dr.vp.yMin);
    vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy };
    setVpVer(v => v + 1);
  }, [points]);
  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);

  const resetViewport = useCallback(() => {
    vpRef.current = { xMin: -0.08, xMax: 1.08, yMin: -0.2, yMax: 3.7 };
    setVpVer(v => v + 1);
  }, []);
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen();
    else scatterRef.current?.requestFullscreen();
  }, []);

  const pushMetrics = useCallback((m: number, b: number) => {
    onMetricsUpdate(computeMetrics(points, m, b));
  }, [points, onMetricsUpdate]);

  /* Training loop — cumulative loss, fixed x-axis */
  useEffect(() => {
    if (!isTraining || points.length < 2) return;

    let m = trained ? weights.m : (Math.random() - 0.5) * 2;
    let b = trained ? weights.b : Math.random();
    const prevLoss = [...lossHistory];
    const target = prevLoss.length + epochs;
    setEpochTarget(target);

    let epoch = 0;
    let animId = 0;

    const step = () => {
      const stepsPerFrame = Math.max(1, Math.floor(epochs / 120));
      for (let s = 0; s < stepsPerFrame && epoch < epochs; s++, epoch++) {
        let dm = 0, db = 0;
        const n = points.length;
        for (const p of points) {
          const err = (m * p.x + b) - p.y;
          dm += (2 / n) * err * p.x;
          db += (2 / n) * err;
        }
        m -= lr * dm; b -= lr * db;
        const mse = points.reduce((acc, p) => acc + ((m * p.x + b) - p.y) ** 2, 0) / n;
        prevLoss.push(mse);
      }
      setWeights({ m, b });
      setLossHistory([...prevLoss]);
      pushMetrics(m, b);

      if (epoch < epochs) { animId = requestAnimationFrame(step); }
      else { setTrained(true); onTrainingComplete(); }
    };
    animId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(animId);
  }, [isTraining]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { pushMetrics(weights.m, weights.b); }, [points]); // eslint-disable-line

  /* Manual weight change */
  const handleManualWeight = useCallback((key: 'm' | 'b', val: number) => {
    setWeights(prev => {
      const next = { ...prev, [key]: val };
      pushMetrics(next.m, next.b);
      return next;
    });
  }, [pushMetrics]);

  /* Equation edit submit */
  const submitEquation = useCallback(() => {
    const m = parseFloat(editM);
    const b = parseFloat(editB);
    if (!isNaN(m) && !isNaN(b)) {
      setWeights({ m, b }); pushMetrics(m, b);
    }
    setEditingEq(false);
  }, [editM, editB, pushMetrics]);

  /* Inference */
  const handleInfer = useCallback(() => {
    const x = parseFloat(inferX);
    if (isNaN(x)) return;
    const y = weights.m * x + weights.b;
    setInferResults(prev => [{ x, y }, ...prev].slice(0, 8));
  }, [inferX, weights]);

  /* Data stats */
  const stats = (() => {
    if (points.length < 2) return null;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    let cov = 0, sx = 0, sy = 0;
    for (let i = 0; i < points.length; i++) {
      cov += (xs[i] - mx) * (ys[i] - my);
      sx += (xs[i] - mx) ** 2; sy += (ys[i] - my) ** 2;
    }
    const r = (sx > 0 && sy > 0) ? cov / Math.sqrt(sx * sy) : 0;
    return { n: points.length, mx, my, xMin: Math.min(...xs), xMax: Math.max(...xs), yMin: Math.min(...ys), yMax: Math.max(...ys), r };
  })();

  /* ─── Render Data Zone ─── */
  useEffect(() => {
    const canvas = dataCanvasRef.current;
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
      const tertiary = root.getPropertyValue('--c-tertiary').trim() || '#e7c365';
      const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
      const border = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';
      ctx.clearRect(0, 0, w, h);

      const padL = 44, padR = 12, padT = 12, padB = 28;
      const dw = w - padL - padR, dh = h - padT - padB;
      const vp = vpRef.current;
      const mapX = (x: number) => padL + ((x - vp.xMin) / (vp.xMax - vp.xMin)) * dw;
      const mapY = (y: number) => padT + ((vp.yMax - y) / (vp.yMax - vp.yMin)) * dh;

      // Compute nice tick steps
      const niceStep = (range: number) => {
        const raw = range / 5;
        const mag = Math.pow(10, Math.floor(Math.log10(raw)));
        const n = raw / mag;
        return (n <= 1.5 ? 1 : n <= 3 ? 2 : n <= 7 ? 5 : 10) * mag;
      };
      const xStep = niceStep(vp.xMax - vp.xMin);
      const yStep = niceStep(vp.yMax - vp.yMin);

      // Grid + tick labels
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.textAlign = 'center';

      // X-axis ticks
      const xStart = Math.ceil(vp.xMin / xStep) * xStep;
      for (let v = xStart; v <= vp.xMax; v += xStep) {
        const px = mapX(v);
        if (px < padL || px > w - padR) continue;
        ctx.strokeStyle = border; ctx.lineWidth = 0.4;
        ctx.beginPath(); ctx.moveTo(px, padT); ctx.lineTo(px, h - padB); ctx.stroke();
        ctx.fillStyle = muted; ctx.globalAlpha = 0.6;
        ctx.fillText(v.toFixed(xStep < 0.1 ? 2 : 1), px, h - padB + 14);
        ctx.globalAlpha = 1;
      }

      // Y-axis ticks
      ctx.textAlign = 'right';
      const yStart = Math.ceil(vp.yMin / yStep) * yStep;
      for (let v = yStart; v <= vp.yMax; v += yStep) {
        const py = mapY(v);
        if (py < padT || py > h - padB) continue;
        ctx.strokeStyle = border; ctx.lineWidth = 0.4;
        ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(w - padR, py); ctx.stroke();
        ctx.fillStyle = muted; ctx.globalAlpha = 0.6;
        ctx.fillText(v.toFixed(yStep < 0.1 ? 2 : 1), padL - 4, py + 3);
        ctx.globalAlpha = 1;
      }

      // Axis lines
      ctx.strokeStyle = `${muted}40`; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

      // Clip data area
      ctx.save();
      ctx.beginPath(); ctx.rect(padL, padT, dw, dh); ctx.clip();

      // Regression line
      const { m, b: intercept } = weights;
      const lx0 = vp.xMin - 1, lx1 = vp.xMax + 1;
      ctx.beginPath(); ctx.moveTo(mapX(lx0), mapY(m * lx0 + intercept)); ctx.lineTo(mapX(lx1), mapY(m * lx1 + intercept));
      ctx.strokeStyle = primary; ctx.lineWidth = 2; ctx.stroke();

      // Residuals
      if (showResiduals) {
        ctx.setLineDash([3, 3]); ctx.lineWidth = 0.8;
        for (const p of points) {
          ctx.beginPath(); ctx.moveTo(mapX(p.x), mapY(p.y)); ctx.lineTo(mapX(p.x), mapY(m * p.x + intercept));
          ctx.strokeStyle = `${tertiary}80`; ctx.stroke();
        }
        ctx.setLineDash([]);
      }

      // Data points
      for (const p of points) {
        const sx = mapX(p.x), sy = mapY(p.y);
        ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = tertiary; ctx.globalAlpha = 0.85; ctx.fill(); ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.strokeStyle = `${tertiary}40`; ctx.lineWidth = 0.5; ctx.stroke();
      }

      // Inference markers
      for (const ir of inferResults) {
        const sx = mapX(ir.x), sy = mapY(ir.y);
        ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ff6b6b'; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(sx, sy, 9, 0, Math.PI * 2);
        ctx.strokeStyle = '#ff6b6b60'; ctx.lineWidth = 1; ctx.stroke();
        ctx.setLineDash([2, 3]); ctx.strokeStyle = '#ff6b6b40'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(sx, h - padB); ctx.lineTo(sx, sy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(padL, sy); ctx.lineTo(sx, sy); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore(); // end clip

      if (dataset === 'custom' && points.length < 3) {
        ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif";
        ctx.globalAlpha = 0.5; ctx.textAlign = 'center';
        ctx.fillText('Click to add data points', w / 2, h / 2);
        ctx.globalAlpha = 1; ctx.textAlign = 'start';
      }
    };
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [points, weights, showResiduals, dataset, inferResults, vpVer]);

  /* ─── Render Loss Curve (fixed x-axis) ─── */
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

      // X-axis ticks — fixed to totalEp
      for (let i = 0; i <= 4; i++) {
        const ep = Math.round((totalEp / 4) * i);
        const x = padL + (i / 4) * cw;
        ctx.fillText(String(ep), x - 6, h - 8);
      }
      ctx.globalAlpha = 1;

      // Loss line — x mapped to fixed totalEp range
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

  /* ─── Render Line Preview (mini canvas in controls section) ─── */
  useEffect(() => {
    const canvas = previewRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
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

    // Background grid
    ctx.strokeStyle = border; ctx.lineWidth = 0.3;
    for (let x = 0; x < w; x += 20) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let y = 0; y < h; y += 20) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

    // Axes
    const padL = 4, padB = 4;
    const mapPX = (x: number) => padL + x * (w - padL * 2);
    const mapPY = (y: number) => h - padB - (y / 3.5) * (h - padB * 2);

    ctx.strokeStyle = `${muted}30`; ctx.lineWidth = 0.6;
    ctx.beginPath(); ctx.moveTo(padL, 0); ctx.lineTo(padL, h - padB); ctx.lineTo(w, h - padB); ctx.stroke();

    // Line
    const { m, b: intercept } = weights;
    ctx.beginPath(); ctx.moveTo(mapPX(0), mapPY(intercept)); ctx.lineTo(mapPX(1), mapPY(m + intercept));
    ctx.strokeStyle = primary; ctx.lineWidth = 2; ctx.stroke();

    // Mini data points
    for (const p of points) {
      ctx.beginPath(); ctx.arc(mapPX(p.x), mapPY(p.y), 2, 0, Math.PI * 2);
      ctx.fillStyle = `${muted}60`; ctx.fill();
    }
  }, [weights, points]);

  const sign = weights.b >= 0 ? '+' : '−';
  const eqStr = `y = ${weights.m.toFixed(3)}x ${sign} ${Math.abs(weights.b).toFixed(3)}`;

  return (
    <div className="viz-scroll">
      {/* Scatter Plot */}
      <div className="viz-scroll__section viz-scroll__section--canvas" ref={scatterRef}>
        <canvas
          ref={dataCanvasRef}
          onClick={dataset === 'custom' ? handleDataClick : undefined}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { dragRef.current = null; setHoverPt(null); }}
          style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }}
        />
        {/* Hover tooltip */}
        {hoverPt && (
          <div className="viz-tooltip" style={{ left: hoverPt.px + 12, top: hoverPt.py - 8 }}>
            ({hoverPt.x.toFixed(3)}, {hoverPt.y.toFixed(3)})
          </div>
        )}
        {/* Scatter controls overlay */}
        <div className="viz-scatter-ctrls">
          <button className="viz-scatter-btn" onClick={resetViewport} title="Reset view">⟲</button>
          <button className="viz-scatter-btn" onClick={() => {
            const vp = vpRef.current; const cx = (vp.xMin+vp.xMax)/2, cy = (vp.yMin+vp.yMax)/2;
            vpRef.current = { xMin: cx+(vp.xMin-cx)*0.8, xMax: cx+(vp.xMax-cx)*0.8, yMin: cy+(vp.yMin-cy)*0.8, yMax: cy+(vp.yMax-cy)*0.8 };
            setVpVer(v=>v+1);
          }} title="Zoom in">+</button>
          <button className="viz-scatter-btn" onClick={() => {
            const vp = vpRef.current; const cx = (vp.xMin+vp.xMax)/2, cy = (vp.yMin+vp.yMax)/2;
            vpRef.current = { xMin: cx+(vp.xMin-cx)*1.25, xMax: cx+(vp.xMax-cx)*1.25, yMin: cy+(vp.yMin-cy)*1.25, yMax: cy+(vp.yMax-cy)*1.25 };
            setVpVer(v=>v+1);
          }} title="Zoom out">−</button>
          <button className="viz-scatter-btn" onClick={toggleFullscreen} title="Fullscreen">⛶</button>
        </div>
      </div>

      {/* Loss Curve */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={lossCanvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* Manual Line Control — with preview on left */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">LINE CONTROL</span>
          <span className="viz-ctrl__subtitle">Adjust weights manually</span>
        </div>
        <div className="viz-ctrl__split">
          {/* Mini preview */}
          <div className="viz-ctrl__preview">
            <canvas ref={previewRef} style={{ width: '100%', height: '100%', display: 'block', borderRadius: '4px' }} />
          </div>
          {/* Controls */}
          <div className="viz-ctrl__right">
            {/* Equation — double-click to edit */}
            <div className="viz-ctrl__equation-wrap">
              {editingEq ? (
                <div className="viz-ctrl__eq-edit">
                  <span>y =</span>
                  <input className="viz-ctrl__eq-input" value={editM} onChange={e => setEditM(e.target.value)} autoFocus onKeyDown={e => e.key === 'Enter' && submitEquation()} />
                  <span>x +</span>
                  <input className="viz-ctrl__eq-input" value={editB} onChange={e => setEditB(e.target.value)} onKeyDown={e => e.key === 'Enter' && submitEquation()} />
                  <button className="viz-ctrl__eq-ok" onClick={submitEquation}>✓</button>
                </div>
              ) : (
                <div className="viz-ctrl__equation" onDoubleClick={() => { setEditM(weights.m.toFixed(3)); setEditB(weights.b.toFixed(3)); setEditingEq(true); }} title="Double-click to edit">
                  {eqStr}
                  <span className="viz-ctrl__eq-hint">dbl-click</span>
                </div>
              )}
            </div>
            <div className="viz-ctrl__sliders">
              <div className="viz-ctrl__slider-row">
                <label>Slope (m)</label>
                <span className="viz-ctrl__slider-val">{weights.m.toFixed(3)}</span>
              </div>
              <input type="range" className="control__range" min={-5} max={5} step={0.01} value={weights.m}
                onChange={e => handleManualWeight('m', Number(e.target.value))} />
              <div className="control__range-labels"><span>-5</span><span>5</span></div>
              <div className="viz-ctrl__slider-row">
                <label>Intercept (b)</label>
                <span className="viz-ctrl__slider-val">{weights.b.toFixed(3)}</span>
              </div>
              <input type="range" className="control__range" min={-3} max={3} step={0.01} value={weights.b}
                onChange={e => handleManualWeight('b', Number(e.target.value))} />
              <div className="control__range-labels"><span>-3</span><span>3</span></div>
            </div>
          </div>
        </div>
      </div>

      {/* Inference */}
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

      {/* Data Statistics */}
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
