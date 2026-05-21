import { usePersistentState } from '../../hooks/usePersistentState';
import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps, MetricValue } from '../registry';
import { usePlayground } from '../../store';

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

  const [points, setPoints] = usePersistentState<Point[]>('omodels-linear-regression-points', []);
  const [weights, setWeights] = useState({ m: 0.5, b: 0.5 });
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [epochTarget, setEpochTarget] = usePersistentState('omodels-linear-regression-epochTarget', 0);
  const [trained, setTrained] = usePersistentState('omodels-linear-regression-trained', false);
  
  // Real-time backpropagation tracker states
  const [slowMode, setSlowMode] = useState(false);
  const [gradients, setGradients] = useState({ dm: 0, db: 0 });
  const [gradMHistory, setGradMHistory] = useState<number[]>([]);
  const [gradBHistory, setGradBHistory] = useState<number[]>([]);
  const [weightMHistory, setWeightMHistory] = useState<number[]>([]);
  const [weightBHistory, setWeightBHistory] = useState<number[]>([]);

  // Viewport: pan/zoom
  const vpRef = useRef({ xMin: -0.08, xMax: 1.08, yMin: -0.2, yMax: 3.7 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);

  // Hover tooltip
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);

  // Inference
  const [inferX, setInferX] = useState('0.50');
  const [inferResults, setInferResults] = usePersistentState<{ x: number; y: number }[]>('omodels-linear-regression-inferResults', []);

  // Equation editing
  const [editingEq, setEditingEq] = useState(false);
  const [editM, setEditM] = useState('');
  const [editB, setEditB] = useState('');

  // Import from store
  const { importedData, importVersion, testData, testVersion, setTestResults } = usePlayground();

  // Test dataset evaluation
  useEffect(() => {
    if (!testData || testData.length === 0) return;
    const total = testData.length;
    const results: Record<string, any> = { total, predictions: [] };

    let ssTot = 0, ssRes = 0, mse = 0, mae = 0, correctRounded = 0;
    const yMean = testData.reduce((s, p) => s + (p.y !== undefined ? p.y : (p.label ?? 0)), 0) / total;
    for (const p of testData) {
      const x = p.x !== undefined ? p.x : (p.features?.[0] ?? 0);
      const y = p.y !== undefined ? p.y : (p.label ?? p.cls ?? 0);
      const yHat = weights.m * x + weights.b;
      const err = y - yHat;
      ssRes += err * err;
      ssTot += (y - yMean) ** 2;
      mse += err * err;
      mae += Math.abs(err);
      if (Math.round(yHat) === Math.round(y)) correctRounded++;
      results.predictions.push({ features: [x], actual: y, predicted: yHat });
    }
    mse /= total;
    mae /= total;
    results.type = 'regression';
    results.r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
    results.mse = mse;
    results.rmse = Math.sqrt(mse);
    results.mae = mae;
    results.accuracy = correctRounded / total;

    setTestResults(results);
  }, [testVersion, testData, weights]); // eslint-disable-line react-hooks/exhaustive-deps

  const metrics = useMemo(() => {
    if (points.length < 2) return { r2: 0, mse: 0, mae: 0, rmse: 0 };
    const yMean = points.reduce((s, p) => s + p.y, 0) / points.length;
    let ssTot = 0, ssRes = 0, mse = 0, mae = 0;
    for (const p of points) {
      const pred = weights.m * p.x + weights.b;
      const err = p.y - pred;
      ssRes += err * err;
      ssTot += (p.y - yMean) ** 2;
      mse += err * err;
      mae += Math.abs(err);
    }
    mse /= points.length;
    mae /= points.length;
    const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
    return { r2, mse, mae, rmse: Math.sqrt(mse) };
  }, [points, weights]);

  const lr = (params.learningRate as number) ?? 0.01;
  const epochs = (params.epochs as number) ?? 100;
  const showResiduals = (params.showResiduals as boolean) ?? false;
  const numPoints = (datasetParams.points as number) ?? 50;
  const noise = (datasetParams.noise as number) ?? 0.3;

  /* Auto-zoom viewport to fit data */
  const autoZoomToFit = useCallback((pts: Point[]) => {
    if (pts.length === 0) return;
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.1 || 0.5;
    const yPad = (yMax - yMin) * 0.1 || 0.5;
    vpRef.current = {
      xMin: xMin - xPad, xMax: xMax + xPad,
      yMin: yMin - yPad, yMax: yMax + yPad,
    };
    setVpVer(v => v + 1);
  }, []);

  /* Generate dataset */
  useEffect(() => {
    if (dataset === 'custom' || dataset === 'import') return;
    const pts = generateData(dataset, numPoints, noise);
    setPoints(pts);
    setWeights({ m: 0.5, b: 0.5 });
    setLossHistory([]); setEpochTarget(0);
    setTrained(false); setInferResults([]);
    vpRef.current = { xMin: -0.08, xMax: 1.08, yMin: -0.2, yMax: 3.7 };
    setVpVer(v => v + 1);
    setGradients({ dm: 0, db: 0 });
    setGradMHistory([]); setGradBHistory([]);
    setWeightMHistory([]); setWeightBHistory([]);
  }, [dataset, numPoints, noise]);

  /* When importedData arrives from store, use it */
  useEffect(() => {
    if (dataset !== 'import' || !importedData || importedData.length === 0) return;
    const pts = importedData as Point[];
    setPoints(pts);
    setWeights({ m: 0.5, b: 0.5 });
    setLossHistory([]); setEpochTarget(0);
    setTrained(false); setInferResults([]);
    autoZoomToFit(pts);
    setGradients({ dm: 0, db: 0 });
    setGradMHistory([]); setGradBHistory([]);
    setWeightMHistory([]); setWeightBHistory([]);
  }, [importVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  /* Full reset (triggered by store resetVersion) */
  useEffect(() => {
    if (resetVersion === 0) return;
    setWeights({ m: 0.5, b: 0.5 });
    setLossHistory([]); setEpochTarget(0);
    setTrained(false); setInferResults([]);
    vpRef.current = { xMin: -0.08, xMax: 1.08, yMin: -0.2, yMax: 3.7 };
    setVpVer(v => v + 1);
    setHoverPt(null); setEditingEq(false);
    setGradients({ dm: 0, db: 0 });
    setGradMHistory([]); setGradBHistory([]);
    setWeightMHistory([]); setWeightBHistory([]);
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

  /* Viewport: wheel zoom — requires Ctrl */
  const handleWheel = useCallback((e: React.WheelEvent) => {
    if (!e.ctrlKey) return; // let normal scroll pass through
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

  /* Viewport: drag pan — requires Ctrl held */
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
     // only pan when Ctrl is held
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
    // Drag (only when Ctrl was held at mousedown)
    const dr = dragRef.current;
    if (!dr) return;
    const dx = ((e.clientX - dr.sx) / rect.width) * (dr.vp.xMax - dr.vp.xMin);
    const dy = ((e.clientY - dr.sy) / rect.height) * (dr.vp.yMax - dr.vp.yMin);
    vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy };
    setVpVer(v => v + 1);
  }, [points]);
  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);

  const resetView = useCallback(() => {
    if (dataset === 'import' && points.length > 1) {
      const xs = points.map(p => p.x), ys = points.map(p => p.y);
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const xPad = (xMax - xMin) * 0.15 || 0.5, yPad = (yMax - yMin) * 0.15 || 0.5;
      vpRef.current = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
    } else {
      vpRef.current = { xMin: -0.08, xMax: 1.08, yMin: -0.2, yMax: 3.7 };
    }
    setVpVer(v => v + 1);
  }, [dataset, points]);

  const zoomBtn = useCallback((f: number) => {
    const vp = vpRef.current;
    const mx = (vp.xMin + vp.xMax) / 2;
    const my = (vp.yMin + vp.yMax) / 2;
    vpRef.current = {
      xMin: mx + (vp.xMin - mx) * f,
      xMax: mx + (vp.xMax - mx) * f,
      yMin: my + (vp.yMin - my) * f,
      yMax: my + (vp.yMax - my) * f
    };
    setVpVer(v => v + 1);
  }, []);


  const pushMetrics = useCallback((m: number, b: number) => {
    onMetricsUpdate(computeMetrics(points, m, b));
  }, [points, onMetricsUpdate]);

  /* Training loop — with z-score normalization for numerical stability */
  useEffect(() => {
    if (!isTraining || points.length < 2) return;

    const n = points.length;

    // Compute normalization stats (z-score)
    const xArr = points.map(p => p.x), yArr = points.map(p => p.y);
    const xMean = xArr.reduce((a, b) => a + b, 0) / n;
    const yMean = yArr.reduce((a, b) => a + b, 0) / n;
    const xStd = Math.sqrt(xArr.reduce((a, x) => a + (x - xMean) ** 2, 0) / n) || 1;
    const yStd = Math.sqrt(yArr.reduce((a, y) => a + (y - yMean) ** 2, 0) / n) || 1;

    // Normalized points
    const norm = points.map(p => ({ x: (p.x - xMean) / xStd, y: (p.y - yMean) / yStd }));

    // Convert existing weights to normalized space for continuation, or init fresh
    let mN: number, bN: number;
    if (trained) {
      // reverse: y = m*x + b  =>  (yN*yStd+yMean) = m*(xN*xStd+xMean)+b
      //  => yN = (m*xStd/yStd)*xN + (m*xMean+b-yMean)/yStd
      mN = weights.m * xStd / yStd;
      bN = (weights.m * xMean + weights.b - yMean) / yStd;
    } else {
      mN = (Math.random() - 0.5) * 0.5;
      bN = (Math.random() - 0.5) * 0.5;
    }

    const prevLoss = [...lossHistory];
    const target = prevLoss.length + epochs;
    setEpochTarget(target);

    let epoch = 0;
    let animId = 0;
    let timeoutId: any = null;
    let diverged = false;

    // Convert normalized weights back to original space
    const toOriginal = (m_n: number, b_n: number) => {
      const mOrig = m_n * yStd / xStd;
      const bOrig = yMean + b_n * yStd - mOrig * xMean;
      return { m: mOrig, b: bOrig };
    };

    const GRAD_CLIP = 5.0;

    const step = () => {
      const stepsPerFrame = slowMode ? 1 : Math.max(1, Math.floor(epochs / 120));
      let lastDm = 0, lastDb = 0;
      for (let s = 0; s < stepsPerFrame && epoch < epochs; s++, epoch++) {
        let dm = 0, db = 0;
        for (const p of norm) {
          const err = (mN * p.x + bN) - p.y;
          dm += (2 / n) * err * p.x;
          db += (2 / n) * err;
        }

        // Gradient clipping
        const gNorm = Math.sqrt(dm * dm + db * db);
        if (gNorm > GRAD_CLIP) {
          const scale = GRAD_CLIP / gNorm;
          dm *= scale;
          db *= scale;
        }

        mN -= lr * dm;
        bN -= lr * db;
        lastDm = dm;
        lastDb = db;

        // NaN / Infinity early stop
        if (!isFinite(mN) || !isFinite(bN)) {
          diverged = true;
          break;
        }

        // Compute MSE in original space for the loss curve
        const orig = toOriginal(mN, bN);
        const mse = points.reduce((acc, p) => acc + ((orig.m * p.x + orig.b) - p.y) ** 2, 0) / n;
        prevLoss.push(isFinite(mse) ? mse : prevLoss[prevLoss.length - 1] ?? 0);
      }

      if (diverged) {
        // Revert to a safe state
        const safeOrig = toOriginal(0, 0);
        setWeights(safeOrig);
        setLossHistory([...prevLoss]);
        pushMetrics(safeOrig.m, safeOrig.b);
        setTrained(false);
        onTrainingComplete();
        return;
      }

      const orig = toOriginal(mN, bN);
      setWeights(orig);
      setLossHistory([...prevLoss]);
      pushMetrics(orig.m, orig.b);

      // Track backpropagation stats in real-time
      setGradients({ dm: lastDm, db: lastDb });
      setGradMHistory(prev => [...prev, lastDm].slice(-50));
      setGradBHistory(prev => [...prev, lastDb].slice(-50));
      setWeightMHistory(prev => [...prev, orig.m].slice(-50));
      setWeightBHistory(prev => [...prev, orig.b].slice(-50));

      if (epoch < epochs) { 
        if (slowMode) {
          timeoutId = setTimeout(step, 150);
        } else {
          animId = requestAnimationFrame(step);
        }
      } else { 
        setTrained(true); 
        onTrainingComplete(); 
      }
    };

    if (slowMode) {
      timeoutId = setTimeout(step, 150);
    } else {
      animId = requestAnimationFrame(step);
    }

    return () => {
      cancelAnimationFrame(animId);
      clearTimeout(timeoutId);
    };
  }, [isTraining, slowMode]); // eslint-disable-line react-hooks/exhaustive-deps

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

      // Draw test data points (if present)
      if (testData && testData.length > 0) {
        for (const p of testData) {
          const tx = p.x !== undefined ? p.x : (p.features?.[0] ?? 0.5);
          const ty = p.y !== undefined ? p.y : (p.label ?? 0.5);
          const sx = mapX(tx), sy = mapY(ty);
          ctx.beginPath(); ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = '#10b981'; ctx.globalAlpha = 0.6; ctx.fill(); ctx.globalAlpha = 1;
          ctx.beginPath(); ctx.arc(sx, sy, 5.5, 0, Math.PI * 2);
          ctx.strokeStyle = '#10b98140'; ctx.lineWidth = 0.5; ctx.stroke();
        }
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
  }, [points, weights, showResiduals, dataset, inferResults, vpVer, testData]);

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

      {/* BACKPROPAGATION TRACKER */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span className="viz-ctrl__title">BACKPROPAGATION TRACKER</span>
            <span className="viz-ctrl__subtitle">Real-time gradient descent flow & math equations</span>
          </div>
          <button 
            onClick={() => setSlowMode(!slowMode)}
            style={{ 
              padding: '6px 12px', fontSize: '10px', fontWeight: 'bold', borderRadius: '16px', cursor: 'pointer', border: 'none',
              background: slowMode ? 'var(--c-primary)' : 'var(--c-surface-variant)', 
              color: slowMode ? '#fff' : 'var(--c-on-surface)' 
            }}
          >
            {slowMode ? '🐢 SLOW TRAINING: ON' : '🐢 SLOW TRAINING: OFF'}
          </button>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '12px' }}>
          {/* Mathematical equations breakdown */}
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', color: 'var(--c-on-surface-variant)', background: 'rgba(0,0,0,0.2)', padding: '10px', overflowX: 'auto', whiteSpace: 'nowrap' }}>
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Linear Regression Formulation</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Model Prediction: <span style={{ color: '#a855f7' }}>ŷ = m·x + b</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Objective (MSE):   <span style={{ color: '#a855f7' }}>E = (1/n) · Σ (ŷᵢ - yᵢ)²</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>2. Backpropagation (Partial Derivatives)</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Slope Gradient (∂E/∂m):     <span style={{ color: 'var(--c-error)' }}>∂E/∂m = (2/n) · Σ (ŷᵢ - yᵢ) · xᵢ</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Intercept Gradient (∂E/∂b): <span style={{ color: 'var(--c-error)' }}>∂E/∂b = (2/n) · Σ (ŷᵢ - yᵢ)</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>3. Parameter Gradient Descent Updates</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Update Slope:     <span style={{ color: 'var(--c-primary)' }}>m ← m - η · (∂E/∂m)</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Update Intercept: <span style={{ color: 'var(--c-primary)' }}>b ← b - η · (∂E/∂b)</span></div>
          </div>

          {/* Real-time sparklines and stats */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            {/* Slope Parameter tracker */}
            <div style={{ padding: '10px', background: 'var(--c-surface-variant)', border: '1px solid var(--c-panel-border)' }}>
              <div style={{ fontWeight: 'bold', fontSize: '12px', color: 'var(--c-primary)', marginBottom: '8px' }}>Slope Parameter (m)</div>
              
              <div style={{ marginBottom: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', opacity: 0.8, marginBottom: '2px' }}>
                  <span>Gradient (∂E/∂m)</span>
                  <span style={{ color: 'var(--c-error)', fontFamily: 'monospace' }}>{gradients.dm.toFixed(4)}</span>
                </div>
                <div style={{ height: '28px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                  {gradMHistory.length > 1 ? (
                    <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                      <line x1="0" y1="12" x2="100" y2="12" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
                      <path 
                        d={gradMHistory.map((v, idx) => {
                          const x = (idx / (gradMHistory.length - 1)) * 100;
                          const max = Math.max(...gradMHistory.map(Math.abs), 0.01);
                          const y = 12 - (v / max) * 12;
                          return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }).join(' ')} 
                        fill="none" stroke="var(--c-error)" strokeWidth="1.5" strokeLinejoin="round" 
                      />
                    </svg>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', fontSize: '9px', opacity: 0.3 }}>No history</div>
                  )}
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', opacity: 0.8, marginBottom: '2px' }}>
                  <span>Weight Value (m)</span>
                  <span style={{ color: 'var(--c-primary)', fontFamily: 'monospace' }}>{weights.m.toFixed(4)}</span>
                </div>
                <div style={{ height: '28px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                  {weightMHistory.length > 1 ? (
                    <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                      <line x1="0" y1="12" x2="100" y2="12" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
                      <path 
                        d={weightMHistory.map((v, idx) => {
                          const x = (idx / (weightMHistory.length - 1)) * 100;
                          const max = Math.max(...weightMHistory.map(Math.abs), 1);
                          const y = 12 - (v / max) * 12;
                          return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }).join(' ')} 
                        fill="none" stroke="var(--c-primary)" strokeWidth="1.5" strokeLinejoin="round" 
                      />
                    </svg>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', fontSize: '9px', opacity: 0.3 }}>No history</div>
                  )}
                </div>
              </div>
            </div>

            {/* Intercept Parameter tracker */}
            <div style={{ padding: '10px', background: 'var(--c-surface-variant)', border: '1px solid var(--c-panel-border)' }}>
              <div style={{ fontWeight: 'bold', fontSize: '12px', color: 'var(--c-tertiary)', marginBottom: '8px' }}>Intercept Parameter (b)</div>
              
              <div style={{ marginBottom: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', opacity: 0.8, marginBottom: '2px' }}>
                  <span>Gradient (∂E/∂b)</span>
                  <span style={{ color: 'var(--c-error)', fontFamily: 'monospace' }}>{gradients.db.toFixed(4)}</span>
                </div>
                <div style={{ height: '28px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                  {gradBHistory.length > 1 ? (
                    <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                      <line x1="0" y1="12" x2="100" y2="12" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
                      <path 
                        d={gradBHistory.map((v, idx) => {
                          const x = (idx / (gradBHistory.length - 1)) * 100;
                          const max = Math.max(...gradBHistory.map(Math.abs), 0.01);
                          const y = 12 - (v / max) * 12;
                          return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }).join(' ')} 
                        fill="none" stroke="var(--c-error)" strokeWidth="1.5" strokeLinejoin="round" 
                      />
                    </svg>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', fontSize: '9px', opacity: 0.3 }}>No history</div>
                  )}
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', opacity: 0.8, marginBottom: '2px' }}>
                  <span>Weight Value (b)</span>
                  <span style={{ color: 'var(--c-primary)', fontFamily: 'monospace' }}>{weights.b.toFixed(4)}</span>
                </div>
                <div style={{ height: '28px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                  {weightBHistory.length > 1 ? (
                    <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                      <line x1="0" y1="12" x2="100" y2="12" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
                      <path 
                        d={weightBHistory.map((v, idx) => {
                          const x = (idx / (weightBHistory.length - 1)) * 100;
                          const max = Math.max(...weightBHistory.map(Math.abs), 1);
                          const y = 12 - (v / max) * 12;
                          return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                        }).join(' ')} 
                        fill="none" stroke="var(--c-primary)" strokeWidth="1.5" strokeLinejoin="round" 
                      />
                    </svg>
                  ) : (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', fontSize: '9px', opacity: 0.3 }}>No history</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* REGRESSION PERFORMANCE MATRIX */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">REGRESSION PERFORMANCE MATRIX</span>
          <span className="viz-ctrl__subtitle">Evaluation metrics for model fit</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', textAlign: 'center', fontSize: '11px', marginTop: '10px' }}>
          <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.r2.toFixed(4)}</div>
            <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>R² (Accuracy of Fit)</div>
            <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Proportion of variance explained</div>
          </div>
          <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.mse.toFixed(4)}</div>
            <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>MSE (Mean Squared Error)</div>
            <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Average squared prediction error</div>
          </div>
          <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.mae.toFixed(4)}</div>
            <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>MAE (Mean Absolute Error)</div>
            <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Average magnitude of errors</div>
          </div>
          <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.rmse.toFixed(4)}</div>
            <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>RMSE (Root MSE)</div>
            <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Standard deviation of residuals</div>
          </div>
        </div>
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
