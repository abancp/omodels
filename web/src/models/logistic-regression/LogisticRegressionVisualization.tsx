import { usePersistentState } from '../../hooks/usePersistentState';
import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClassificationData, computeLoss, computeGradients, computeMetrics, predict, formatEquation, computeConfusionMatrix, computeROCCurve, computeDataStats, type Point, type Weights } from './math';
import { usePlayground } from '../../store';

export default function LogisticRegressionVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const rocCanvasRef = useRef<HTMLCanvasElement>(null);

  
  // State
  const [points, setPoints] = usePersistentState<Point[]>('omodels-logistic-regression-points', []);
  const degree = parseInt((params.degree as string) ?? '1', 10);
  const [weights, setWeights] = useState<Weights>([]); 
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [epochTarget, setEpochTarget] = usePersistentState('omodels-logistic-regression-epochTarget', 0);

  // Real-time backpropagation tracker states
  const [slowMode, setSlowMode] = useState(false);
  const [gradients, setGradients] = useState<number[]>([0, 0, 0]);
  const [gradHistories, setGradHistories] = useState<number[][]>([[], [], []]);
  const [weightHistories, setWeightHistories] = useState<number[][]>([[], [], []]);

  // Viewport
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);

  // Inference
  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = usePersistentState<{x: number, y: number, prob: number, cls: number}[]>('omodels-logistic-regression-inferResults', []);

  // Params
  const lr = (params.learningRate as number) ?? 0.1;
  const epochs = (params.epochs as number) ?? 200;
  const threshold = (params.threshold as number) ?? 0.5;
  const regularization = (params.regularization as string) ?? 'none';
  const regStrength = (params.regStrength as number) ?? 0.01;
  const numPoints = (datasetParams.points as number) ?? 100;
  const noise = (datasetParams.noise as number) ?? 0.15;

  const pushMetrics = useCallback((w: Weights) => {
    onMetricsUpdate(computeMetrics(points, w, threshold));
  }, [points, threshold, onMetricsUpdate]);

  /* Init Weights */
  // Import from store
  const { importedData, importVersion, testData, testVersion, setTestResults } = usePlayground();

  // Test dataset evaluation
  useEffect(() => {
    if (!testData || testData.length === 0) return;
    const total = testData.length;
    const results: Record<string, any> = { total, predictions: [] };

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const p of testData) {
      const x = p.x !== undefined ? p.x : (p.features?.[0] ?? 0);
      const y = p.y !== undefined ? p.y : (p.features?.[1] ?? 0);
      const trueClass = p.cls !== undefined ? p.cls : (p.label ?? 0);
      const prob = predict(x, y, weights);
      const predClass = prob >= threshold ? 1 : 0;

      if (trueClass === 1 && predClass === 1) tp++;
      else if (trueClass === 0 && predClass === 0) tn++;
      else if (trueClass === 0 && predClass === 1) fp++;
      else fn++;
      results.predictions.push({ features: [x, y], actual: trueClass, predicted: predClass, confidence: prob });
    }
    results.type = 'binary';
    results.accuracy = (tp + tn) / total;
    results.precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    results.recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    results.f1 = (results.precision + results.recall) > 0 ? 2 * results.precision * results.recall / (results.precision + results.recall) : 0;
    results.tp = tp; results.tn = tn; results.fp = fp; results.fn = fn;
    results.confusionMatrix = [[tn, fp], [fn, tp]];

    setTestResults(results);
  }, [testVersion, testData, weights, threshold]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (dataset !== 'import' || !importedData || importedData.length === 0) return;
    // Clamp cls to binary (0 or 1) — logistic regression is binary only
    const pts = (importedData as Point[]).map(p => ({
      x: p.x, y: p.y, cls: p.cls >= 1 ? 1 : 0,
    }));
    setPoints(pts);
    // Reset model state for clean training
    const requiredLen = degree === 1 ? 3 : 6;
    setWeights(new Array(requiredLen).fill(0));
    setLossHistory([]); setEpochTarget(0);
    setInferResults([]);
    setGradients(new Array(requiredLen).fill(0));
    setGradHistories(new Array(requiredLen).fill([]));
    setWeightHistories(new Array(requiredLen).fill([]));
    // Auto-zoom viewport
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.15 || 0.5, yPad = (yMax - yMin) * 0.15 || 0.5;
    vpRef.current = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
    setVpVer(v => v + 1);
  }, [importVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  /* Handle Degree Change (resize weights) */
  useEffect(() => {
    setWeights(prev => {
      const requiredLen = degree === 1 ? 3 : 6;
      if (prev.length === requiredLen) return prev;
      const next = new Array(requiredLen).fill(0);
      for (let i = 0; i < Math.min(prev.length, requiredLen); i++) next[i] = prev[i];
      return next;
    });
    setGradients(new Array(degree === 1 ? 3 : 6).fill(0));
    setGradHistories(new Array(degree === 1 ? 3 : 6).fill([]));
    setWeightHistories(new Array(degree === 1 ? 3 : 6).fill([]));
  }, [degree]);

  /* Generate dataset */
  useEffect(() => {
    if (dataset === 'custom' || dataset === 'import') return;
    const pts = generateClassificationData(dataset, numPoints, noise);
    setPoints(pts);
    const requiredLen = degree === 1 ? 3 : 6;
    setWeights(new Array(requiredLen).fill(0));
    setLossHistory([]); setEpochTarget(0);
    setGradients(new Array(requiredLen).fill(0));
    setGradHistories(new Array(requiredLen).fill([]));
    setWeightHistories(new Array(requiredLen).fill([]));
  }, [dataset, numPoints, noise, degree]);

  /* Full reset */
  useEffect(() => {
    if (resetVersion === 0) return;
    const requiredLen = degree === 1 ? 3 : 6;
    setWeights(new Array(requiredLen).fill(0));
    setLossHistory([]); setEpochTarget(0);
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1);
    setHoverPt(null);
    setGradients(new Array(requiredLen).fill(0));
    setGradHistories(new Array(requiredLen).fill([]));
    setWeightHistories(new Array(requiredLen).fill([]));
  }, [resetVersion, degree]);

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
    if (dataset === 'import' && points.length > 1) {
      const xs = points.map(p => p.x), ys = points.map(p => p.y);
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const xPad = (xMax - xMin) * 0.15 || 0.1, yPad = (yMax - yMin) * 0.15 || 0.1;
      vpRef.current = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
    } else {
      vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    }
    setVpVer(v => v + 1);
  }, [dataset, points]);

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

    let w = [...weights];
    let epoch = 0;
    let animId = 0;
    let timeoutId: any = null;
    const prevLoss = [...lossHistory];
    const totalTarget = prevLoss.length + epochs;
    setEpochTarget(totalTarget);
    let lastValidW = [...w];

    const step = () => {
      const stepsPerFrame = slowMode ? 1 : Math.max(1, Math.floor(epochs / 60));
      let diverged = false;
      let lastGrads = [...w];
      for (let s = 0; s < stepsPerFrame && epoch < epochs; s++, epoch++) {
        const grads = computeGradients(points, w, regularization, regStrength, degree);
        lastGrads = [...grads];

        // Gradient clipping
        let gradNorm = 0;
        for (let j = 0; j < grads.length; j++) gradNorm += grads[j] * grads[j];
        gradNorm = Math.sqrt(gradNorm);
        const maxNorm = 5.0;
        if (gradNorm > maxNorm) {
          const scale = maxNorm / gradNorm;
          for (let j = 0; j < grads.length; j++) grads[j] *= scale;
        }

        // Update weights
        for (let j = 0; j < w.length; j++) {
          w[j] -= lr * grads[j];
        }

        // NaN/Infinity safety
        if (w.some(v => !isFinite(v))) {
          w = [...lastValidW];
          diverged = true;
          break;
        }
        lastValidW = [...w];

        const loss = computeLoss(points, w, regularization, regStrength, degree);
        prevLoss.push(isFinite(loss) ? loss : prevLoss[prevLoss.length - 1] ?? 0);
      }
      setWeights([...w]);
      setLossHistory([...prevLoss]);
      pushMetrics(w);

      // Tracker state updates
      setGradients(lastGrads);
      setGradHistories(prev => lastGrads.map((g, i) => [...(prev[i] || []), g].slice(-50)));
      setWeightHistories(prev => w.map((v, i) => [...(prev[i] || []), v].slice(-50)));

      if (diverged || epoch >= epochs) { 
        onTrainingComplete(); 
      } else { 
        if (slowMode) {
          timeoutId = setTimeout(step, 150);
        } else {
          animId = requestAnimationFrame(step);
        }
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

  useEffect(() => { pushMetrics(weights); }, [points, threshold]); // eslint-disable-line

  /* Draw Data Canvas */
  useEffect(() => {
    const canvas = dataCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

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
    if (weights.length > 0) {
      const res = 6; 
      for (let px = 0; px < W; px += res) {
        for (let py = 0; py < H; py += res) {
          const nx = vp.xMin + ((px + res/2) / W) * (vp.xMax - vp.xMin);
          const ny = vp.yMax - ((py + res/2) / H) * (vp.yMax - vp.yMin);
          const prob = predict(nx, ny, weights);
          
          if (prob >= threshold) {
            ctx.fillStyle = `${tertiary}15`;
          } else {
            ctx.fillStyle = `${primary}15`;
          }
          ctx.fillRect(px, py, res, res);
        }
      }

      // Draw decision boundary contour
      ctx.beginPath();
      for (let px = 0; px < W; px += 2) {
        for (let py = 0; py < H; py += 2) {
          const nx = vp.xMin + (px / W) * (vp.xMax - vp.xMin);
          const ny = vp.yMax - (py / H) * (vp.yMax - vp.yMin);
          const prob = predict(nx, ny, weights);
          if (Math.abs(prob - threshold) < 0.015) {
            ctx.fillStyle = '#ffffff60';
            ctx.fillRect(px, py, 2, 2);
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

    // Points
    for (const p of points) {
      ctx.beginPath();
      ctx.arc(mapX(p.x), mapY(p.y), 5, 0, Math.PI * 2);
      ctx.fillStyle = p.cls === 0 ? primary : tertiary;
      ctx.fill();
      ctx.strokeStyle = '#00000040';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Inference markers
    for (const ir of inferResults) {
      const sx = mapX(ir.x), sy = mapY(ir.y);
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = ir.cls === 0 ? primary : tertiary; 
      ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(sx, sy, 9, 0, Math.PI * 2);
      ctx.strokeStyle = ir.cls === 0 ? `${primary}60` : `${tertiary}60`; 
      ctx.lineWidth = 1; ctx.stroke();
    }
  }, [points, weights, vpVer, threshold, degree, inferResults]);

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
      const primary = root.getPropertyValue('--c-primary').trim() || '#cfbcff';
      const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
      const border = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';
      ctx.clearRect(0, 0, w, h);

      ctx.fillStyle = muted; ctx.font = "600 10px 'Inter', sans-serif";
      ctx.globalAlpha = 0.6; ctx.fillText('CROSS ENTROPY LOSS', 12, 18); ctx.globalAlpha = 1;

      const totalEp = epochTarget > 0 ? epochTarget : epochs;
      if (lossHistory.length < 1) return;

      const padL = 40, padR = 16, padT = 30, padB = 24;
      const cw = w - padL - padR, ch = h - padT - padB;

      ctx.strokeStyle = border; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

      const maxLoss = Math.max(...lossHistory) * 1.1 || 1;

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

      // Fill
      const lastX = padL + ((lossHistory.length - 1) / (totalEp - 1)) * cw;
      const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
      gradient.addColorStop(0, `${primary}20`); gradient.addColorStop(1, `${primary}02`);
      ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath();
      ctx.fillStyle = gradient; ctx.fill();
    };
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [lossHistory, epochTarget, epochs]);

  /* Draw ROC Curve */
  useEffect(() => {
    const canvas = rocCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);
    
    const root = getComputedStyle(document.documentElement);
    const primary = root.getPropertyValue('--c-primary').trim() || '#cfbcff';
    const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
    const border = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';

    const padL = 30, padB = 20, padT = 10, padR = 10;
    const cw = w - padL - padR, ch = h - padT - padB;

    ctx.strokeStyle = border; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, padT); ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);

    const { curve, auc } = computeROCCurve(points, weights);
    if (curve.length > 0) {
      ctx.beginPath();
      for (let i = 0; i < curve.length; i++) {
        const x = padL + curve[i].fpr * cw;
        const y = h - padB - curve[i].tpr * ch;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = primary; ctx.lineWidth = 2; ctx.stroke();
    }

    ctx.fillStyle = muted; ctx.font = "9px 'Inter', sans-serif";
    ctx.fillText('FPR', w / 2, h - 5);
    ctx.save(); ctx.translate(10, h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('TPR', -10, 0); ctx.restore();
    ctx.fillStyle = primary; ctx.font = "600 10px 'Inter', sans-serif";
    ctx.fillText(`AUC: ${auc.toFixed(3)}`, w - 50, padT + 10);
  }, [points, weights]);

  const handleInfer = useCallback(() => {
    const x = parseFloat(inferX);
    const y = parseFloat(inferY);
    if (isNaN(x) || isNaN(y)) return;
    const prob = predict(x, y, weights);
    const cls = prob >= threshold ? 1 : 0;
    setInferResults(prev => [{x, y, prob, cls}, ...prev].slice(0, 5));
  }, [inferX, inferY, weights, threshold]);

  const stats = useMemo(() => computeDataStats(points), [points]);
  const cm = useMemo(() => computeConfusionMatrix(points, weights, threshold), [points, weights, threshold]);

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

      {/* 3. EVALUATION MATRIX & ROC */}
      <div className="viz-scroll__section viz-scroll__section--controls" style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">CONFUSION MATRIX</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: '4px', textAlign: 'center', fontSize: '11px' }}>
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
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">ROC CURVE</span>
          </div>
          <div style={{ height: '120px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)', overflow: 'hidden' }}>
            <canvas ref={rocCanvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
          </div>
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
              {weights.map((w, i) => {
                const labels = degree === 1 ? ['Bias (w₀)', 'w₁ (x₁)', 'w₂ (x₂)'] : ['Bias (w₀)', 'w₁ (x₁)', 'w₂ (x₂)', 'w₃ (x₁²)', 'w₄ (x₂²)', 'w₅ (x₁x₂)'];
                return (
                  <div key={i} style={{ marginBottom: '8px' }}>
                    <div className="viz-ctrl__slider-row">
                      <label>{labels[i]}</label>
                      <span className="viz-ctrl__slider-val">{w.toFixed(3)}</span>
                    </div>
                    <input
                      type="range" className="control__range"
                      min="-15" max="15" step="0.05"
                      value={w}
                      onChange={(e) => {
                        const next = [...weights];
                        next[i] = parseFloat(e.target.value);
                        setWeights(next);
                      }}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </div>
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
              padding: '6px 12px', fontSize: '10px', fontWeight: 'bold', cursor: 'pointer', border: 'none',
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
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Logistic Regression Formulation</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Model Prediction: <span style={{ color: '#a855f7' }}>ŷ = σ(w₀ + w₁x₁ + w₂x₂) = 1 / (1 + e⁻ᶻ)</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Objective (BCE):   <span style={{ color: '#a855f7' }}>E = -(1/n) · Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>2. Backpropagation (Partial Derivatives via Chain Rule)</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Bias Gradient (∂E/∂w₀):   <span style={{ color: 'var(--c-error)' }}>∂E/∂w₀ = (1/n) · Σ (ŷᵢ - yᵢ)</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Weight Gradient (∂E/∂wⱼ): <span style={{ color: 'var(--c-error)' }}>∂E/∂wⱼ = (1/n) · Σ (ŷᵢ - yᵢ) · xᵢⱼ</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>3. Parameter Gradient Descent Updates</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Update Parameter j: <span style={{ color: 'var(--c-primary)' }}>wⱼ ← wⱼ - η · (∂E/∂wⱼ)</span></div>
          </div>

          {/* Real-time sparklines and stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
            {weights.slice(0, 3).map((w, i) => {
              const labels = ['Bias Parameter (w₀)', 'Weight 1 (w₁ for x₁)', 'Weight 2 (w₂ for x₂)'];
              const gradLabels = ['∂E/∂w₀', '∂E/∂w₁', '∂E/∂w₂'];
              const gradHistory = gradHistories[i] || [];
              const weightHistory = weightHistories[i] || [];
              return (
                <div key={i} style={{ padding: '10px', background: 'var(--c-surface-variant)', border: '1px solid var(--c-panel-border)' }}>
                  <div style={{ fontWeight: 'bold', fontSize: '12px', color: i === 0 ? 'var(--c-tertiary)' : 'var(--c-primary)', marginBottom: '8px' }}>{labels[i]}</div>
                  
                  <div style={{ marginBottom: '10px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', opacity: 0.8, marginBottom: '2px' }}>
                      <span>Gradient ({gradLabels[i]})</span>
                      <span style={{ color: 'var(--c-error)', fontFamily: 'monospace' }}>{(gradients[i] || 0).toFixed(4)}</span>
                    </div>
                    <div style={{ height: '28px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                      {gradHistory.length > 1 ? (
                        <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                          <line x1="0" y1="12" x2="100" y2="12" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
                          <path 
                            d={gradHistory.map((v, idx) => {
                              const x = (idx / (gradHistory.length - 1)) * 100;
                              const max = Math.max(...gradHistory.map(Math.abs), 0.01);
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
                      <span>Weight Value</span>
                      <span style={{ color: 'var(--c-primary)', fontFamily: 'monospace' }}>{w.toFixed(4)}</span>
                    </div>
                    <div style={{ height: '28px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                      {weightHistory.length > 1 ? (
                        <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                          <line x1="0" y1="12" x2="100" y2="12" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" strokeDasharray="2,2" />
                          <path 
                            d={weightHistory.map((v, idx) => {
                              const x = (idx / (weightHistory.length - 1)) * 100;
                              const max = Math.max(...weightHistory.map(Math.abs), 1);
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
              );
            })}
          </div>
        </div>
      </div>

      {/* 5. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">INFERENCE</span>
          <span className="viz-ctrl__subtitle">Predict class for given x₁, x₂</span>
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
            <label>P(y=1)</label>
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
              <span>x₁</span><span>x₂</span><span>P(y=1)</span><span>Class</span>
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

      {/* 6. DATA STATISTICS */}
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">DATA STATISTICS</span>
          </div>
          <div className="viz-stats__grid">
            <div className="viz-stats__item"><span className="viz-stats__label">N</span><span className="viz-stats__val">{stats.n}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 0</span><span className="viz-stats__val viz-stats__val--primary">{stats.nClass0}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 1</span><span className="viz-stats__val" style={{ color: 'var(--c-tertiary)' }}>{stats.nClass1}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Balance</span><span className="viz-stats__val">{(stats.classBalance * 100).toFixed(1)}% Class 1</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₁</span><span className="viz-stats__val">{stats.meanX.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₂</span><span className="viz-stats__val">{stats.meanY.toFixed(3)}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
