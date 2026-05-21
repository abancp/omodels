import { usePersistentState } from '../../hooks/usePersistentState';
import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateData, trainStep, predictPerceptron, type Point, type PerceptronState, type ActivationType } from './math';
import { drawDataCanvas, drawNetworkCanvas } from './drawHelpers';
import { usePlayground } from '../../store';

const FEATURE_COLORS = [
  '#ff4757', // Coral Red
  '#2ed573', // Bright Green
  '#1e90ff', // Neon Blue
  '#ffa502', // Orange
  '#eccc68', // Soft Gold
  '#a855f7', // Purple
  '#00d2d3', // Teal
];

export default function PerceptronVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const netRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const scatterContainerRef = useRef<HTMLDivElement>(null);

  const activation = (params.activation as ActivationType) ?? 'step';
  const outAct = (params.outAct as ActivationType) ?? 'sigmoid';
  const lr = (params.learningRate as number) ?? 0.1;
  const maxEp = (params.maxEpochs as number) ?? 100;
  const inputNodes = (params.inputNodes as number) ?? 2;
  const numPerceptrons = (params.numPerceptrons as number) ?? 1;
  
  const numPoints = (datasetParams.points as number) ?? 100;
  const noise = (datasetParams.noise as number) ?? 0.2;

  const [points, setPoints] = usePersistentState<Point[]>('omodels-perceptron-points', []);
  const [percState, setPercState] = useState<PerceptronState>({
    numInputs: inputNodes,
    numPerceptrons: numPerceptrons,
    hiddenWeights: Array.from({ length: numPerceptrons }, () => Array.from({ length: inputNodes }, () => Math.random() - 0.5)),
    hiddenBias: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
    outWeights: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
    outBias: Math.random() - 0.5,
    activation,
    outAct,
    learningRate: lr,
    epoch: 0,
    maxEpochs: maxEp,
    lossHistory: [],
    converged: false
  });
  
  const [epochTarget, setEpochTarget] = usePersistentState('omodels-perceptron-epochTarget', 0);
  const [trained, setTrained] = usePersistentState('omodels-perceptron-trained', false);

  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current; moved?: boolean } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ features: number[]; px: number; py: number; hoverFeatureIdx?: number; label: number } | null>(null);
  const [xAxisMode, setXAxisMode] = usePersistentState<'standard' | 'multi-feature'>('omodels-perceptron-xAxisMode', 'standard');
  const [selectedXFeatureIdx, setSelectedXFeatureIdx] = usePersistentState<number>('omodels-perceptron-selectedXFeatureIdx', 0);
  const [showDataPreview, setShowDataPreview] = usePersistentState('omodels-perceptron-showDataPreview', false);
  
  // Advanced Mode State
  const [advancedMode, setAdvancedMode] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState<'hidden' | 'output'>('hidden');
  const [selectedNeuron, setSelectedNeuron] = useState<number>(0);

  const [shouldScaleInputs, setShouldScaleInputs] = usePersistentState('omodels-perceptron-shouldScaleInputs', true);

  // Inference
  const [inferInputs, setInferInputs] = usePersistentState<string[]>('omodels-perceptron-inferInputs', Array(inputNodes).fill('0.50'));
  const [inferResults, setInferResults] = usePersistentState<{ features: number[]; pred: number; rawFeatures?: number[] }[]>('omodels-perceptron-inferResults', []);

  // Migrate points schema if needed
  useEffect(() => {
    let changed = false;
    const migrated = points.map((p: any) => {
      if (p && p.features && Array.isArray(p.features)) return p;
      changed = true;
      const x = p.x !== undefined ? p.x : 0.5;
      const y = p.y !== undefined ? p.y : 0.5;
      return {
        features: [x, y],
        label: p.label ?? p.cls ?? 0
      };
    });
    if (changed) {
      setPoints(migrated);
    }
  }, [points, setPoints]);

  const migratedPoints = useMemo(() => {
    return points.map((p: any) => {
      if (p && p.features && Array.isArray(p.features)) return p as Point;
      const x = p.x !== undefined ? p.x : 0.5;
      const y = p.y !== undefined ? p.y : 0.5;
      return {
        features: [x, y],
        label: p.label ?? p.cls ?? 0
      } as Point;
    });
  }, [points]);

  const metrics = useMemo(() => {
    const total = migratedPoints.length;
    let regressionMetrics = { r2: 0, mse: 0, mae: 0, rmse: 0 };
    let binaryMetrics = { tp: 0, tn: 0, fp: 0, fn: 0, acc: 0, prec: 0, rec: 0, f1: 0 };

    if (total === 0) {
      return { regressionMetrics, binaryMetrics };
    }

    if (outAct === 'linear') {
      const yMean = migratedPoints.reduce((acc, p) => acc + p.label, 0) / total;
      let ssTot = 0, ssRes = 0, mse = 0, mae = 0;
      for (const p of migratedPoints) {
        const { pred } = predictPerceptron(p, percState);
        const yHat = pred;
        const y = p.label;
        const err = y - yHat;
        ssRes += err * err;
        ssTot += (y - yMean) ** 2;
        mse += err * err;
        mae += Math.abs(err);
      }
      mse /= total;
      mae /= total;
      const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
      regressionMetrics = { r2, mse, mae, rmse: Math.sqrt(mse) };
    } else {
      let tp = 0, tn = 0, fp = 0, fn = 0;
      for (const p of migratedPoints) {
        const { pred } = predictPerceptron(p, percState);
        const predClass = pred >= 0.5 ? 1 : 0;
        if (p.label === 1 && predClass === 1) tp++;
        else if (p.label === 0 && predClass === 0) tn++;
        else if (p.label === 0 && predClass === 1) fp++;
        else if (p.label === 1 && predClass === 0) fn++;
      }
      const acc = (tp + tn) / total;
      const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
      const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
      const f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
      binaryMetrics = { tp, tn, fp, fn, acc, prec, rec, f1 };
    }

    return { regressionMetrics, binaryMetrics };
  }, [migratedPoints, percState, outAct]);

  const pushMetrics = useCallback((s: PerceptronState) => {
    const loss = s.lossHistory[s.lossHistory.length - 1] ?? 0;

    if (outAct === 'linear') {
      onMetricsUpdate([
        { label: 'Loss (MSE)', value: loss.toFixed(4), isPrimary: true },
        { label: 'R²', value: metrics.regressionMetrics.r2.toFixed(4) },
        { label: 'MAE', value: metrics.regressionMetrics.mae.toFixed(4) },
        { label: 'Epochs', value: String(s.epoch) },
        { label: 'Converged', value: s.converged ? 'Yes' : 'No' },
      ]);
    } else {
      onMetricsUpdate([
        { label: 'Accuracy', value: (metrics.binaryMetrics.acc * 100).toFixed(1) + '%', isPrimary: true },
        { label: 'Loss (MSE)', value: loss.toFixed(4) },
        { label: 'Precision', value: metrics.binaryMetrics.prec.toFixed(3) },
        { label: 'Recall', value: metrics.binaryMetrics.rec.toFixed(3) },
        { label: 'F1 Score', value: metrics.binaryMetrics.f1.toFixed(3) },
        { label: 'Epochs', value: String(s.epoch) },
        { label: 'Converged', value: s.converged ? 'Yes' : 'No' },
      ]);
    }
  }, [outAct, metrics, onMetricsUpdate]);

  const resetState = useCallback(() => {
    setPercState(s => ({
      ...s,
      numInputs: inputNodes,
      numPerceptrons: numPerceptrons,
      hiddenWeights: Array.from({ length: numPerceptrons }, () => Array.from({ length: inputNodes }, () => Math.random() - 0.5)),
      hiddenBias: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
      outWeights: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
      outBias: Math.random() - 0.5,
      epoch: 0,
      lossHistory: [],
      converged: false
    }));
    setEpochTarget(0);
    setTrained(false);
    setInferResults([]);
  }, [inputNodes, numPerceptrons]);

  // Adjust inferInputs array size when inputNodes changes
  // Import from store
  const { importedData, importVersion, importStats, testData, testVersion, setTestResults } = usePlayground();

  // Test dataset evaluation
  useEffect(() => {
    if (!testData || testData.length === 0) return;
    const isRegression = outAct === 'linear';
    const total = testData.length;
    const results: Record<string, any> = { total, predictions: [] };

    if (isRegression) {
      let ssTot = 0, ssRes = 0, mse = 0, mae = 0, correctRounded = 0;
      const yMean = testData.reduce((s, p) => s + (p.label ?? 0), 0) / total;
      for (const p of testData) {
        const feats = p.features || [p.x ?? 0, p.y ?? 0];
        const { pred } = predictPerceptron(feats, percState);
        const yHat = pred;
        const y = p.label ?? 0;
        const err = y - yHat;
        ssRes += err * err;
        ssTot += (y - yMean) ** 2;
        mse += err * err;
        mae += Math.abs(err);
        if (Math.round(yHat) === Math.round(y)) correctRounded++;
        results.predictions.push({ features: feats, actual: y, predicted: yHat });
      }
      mse /= total;
      mae /= total;
      results.type = 'regression';
      results.r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
      results.mse = mse;
      results.rmse = Math.sqrt(mse);
      results.mae = mae;
      results.accuracy = correctRounded / total;
    } else {
      let tp = 0, tn = 0, fp = 0, fn = 0;
      for (const p of testData) {
        const feats = p.features || [p.x ?? 0, p.y ?? 0];
        const { pred } = predictPerceptron(feats, percState);
        const predClass = pred >= 0.5 ? 1 : 0;
        const trueClass = p.label ?? 0;
        if (trueClass === 1 && predClass === 1) tp++;
        else if (trueClass === 0 && predClass === 0) tn++;
        else if (trueClass === 0 && predClass === 1) fp++;
        else fn++;
        results.predictions.push({ features: feats, actual: trueClass, predicted: predClass, confidence: pred });
      }
      results.type = 'binary';
      results.accuracy = (tp + tn) / total;
      results.precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
      results.recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
      results.f1 = (results.precision + results.recall) > 0 ? 2 * results.precision * results.recall / (results.precision + results.recall) : 0;
      results.tp = tp; results.tn = tn; results.fp = fp; results.fn = fn;
      results.confusionMatrix = [[tn, fp], [fn, tp]];
    }
    setTestResults(results);
  }, [testVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (dataset !== 'import' || !importedData || importedData.length === 0) return;
    // Convert {features,label} to Perceptron's {features,label} format, preferring pre-mapped high-dimensional features
    const pts = importedData.map((p: any) => ({
      features: p.features || [p.x, ...(p.y !== undefined ? [p.y] : [])],
      label: p.cls ?? p.label ?? 0,
    }));
    setPoints(pts);
    // Auto-zoom viewport
    const xs = importedData.map((p: any) => p.x), ys = importedData.map((p: any) => p.y ?? 0);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.1 || 0.5, yPad = (yMax - yMin) * 0.1 || 0.5;
    vpRef.current = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
    setVpVer(v => v + 1);
  }, [importVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setInferInputs(prev => {
      if (prev.length === inputNodes) return prev;
      const next = [...prev];
      while (next.length < inputNodes) next.push('0.50');
      return next.slice(0, inputNodes);
    });
    
    setPercState(s => {
      if (s.numInputs === inputNodes && s.numPerceptrons === numPerceptrons) return s;
      setEpochTarget(0);
      setTrained(false);
      return {
        ...s,
        numInputs: inputNodes,
        numPerceptrons: numPerceptrons,
        hiddenWeights: Array.from({ length: numPerceptrons }, () => Array.from({ length: inputNodes }, () => Math.random() - 0.5)),
        hiddenBias: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
        outWeights: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
        outBias: Math.random() - 0.5,
        epoch: 0,
        lossHistory: [],
        converged: false
      };
    });
  }, [inputNodes, numPerceptrons]);

  useEffect(() => { 
    if (dataset === 'custom' || dataset === 'import') {
      if (dataset === 'custom') { setPoints([]); resetState(); }
      return; 
    }
    setPoints(generateData(dataset, numPoints, noise, inputNodes)); 
    resetState();
  }, [dataset, numPoints, noise, inputNodes, numPerceptrons, resetState]);

  useEffect(() => { 
    if (resetVersion === 0) return; 
    resetState();
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; 
    setVpVer(v => v + 1); 
  }, [resetVersion, resetState]);

  useEffect(() => { 
    setPercState(s => ({ ...s, activation, outAct, learningRate: lr, maxEpochs: maxEp })); 
  }, [activation, outAct, lr, maxEp]);

  // Mouse Handlers for Scatter Plot
  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { 
    if (dragRef.current?.moved) return; 
    if (dataset !== 'custom') return;
    const c = dataRef.current; if (!c) return; 
    const r = c.getBoundingClientRect(); const vp = vpRef.current; 
    const padL = 20, padT = 20, padR = 20, padB = 20;
    const mx = e.clientX - r.left; const my = e.clientY - r.top;
    if (mx < padL || mx > r.width - padR || my < padT || my > r.height - padB) return;
    
    const dw = r.width - padL - padR; const dh = r.height - padT - padB;
    const px = vp.xMin + ((mx - padL) / dw) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((my - padT) / dh) * (vp.yMax - vp.yMin);
    
    // Set other features to 0.5
    const features = Array(inputNodes).fill(0.5);
    features[0] = px;
    if (inputNodes > 1) features[1] = py;
    
    setPoints(prev => [...prev, { features, label: e.altKey ? 0 : 1 }]); 
  }, [inputNodes]);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => { 
    if (!e.ctrlKey) return; 
    e.preventDefault(); 
    const f = e.deltaY > 0 ? 1.1 : 0.9; 
    const c = dataRef.current; if (!c) return; 
    const r = c.getBoundingClientRect(); const vp = vpRef.current; 
    const padL = 20, padT = 20, padR = 20, padB = 20;
    const mx = e.clientX - r.left; const my = e.clientY - r.top;
    const dw = r.width - padL - padR; const dh = r.height - padT - padB;
    const px = vp.xMin + ((mx - padL) / dw) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((my - padT) / dh) * (vp.yMax - vp.yMin);
    
    vpRef.current = { 
      xMin: px + (vp.xMin - px) * f, xMax: px + (vp.xMax - px) * f, 
      yMin: py + (vp.yMin - py) * f, yMax: py + (vp.yMax - py) * f 
    }; 
    setVpVer(v => v + 1); 
  }, []);

  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { 
    dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current }, moved: false }; 
  }, []);

  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { 
    const c = dataRef.current; if (!c) return; 
    const r = c.getBoundingClientRect(); const vp = vpRef.current; 
    const padL = 20, padT = 20, padR = 20, padB = 20;
    const mx = e.clientX - r.left; const my = e.clientY - r.top;
    
    // Hover tooltip
    const dw = r.width - padL - padR; const dh = r.height - padT - padB;
    const px = vp.xMin + ((mx - padL) / dw) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((my - padT) / dh) * (vp.yMax - vp.yMin);
    
    let nearest: typeof hoverPt = null;
    let minD = 0.04 * (vp.xMax - vp.xMin); // roughly 4% of viewport width

    // Convert mouse pixels back to standard/multi-feature space
    const sx_val = ((mx - padL) / dw) * (vp.xMax - vp.xMin) + vp.xMin;
    const sy_val = ((dh - (my - padT)) / dh) * (vp.yMax - vp.yMin) + vp.yMin;

    if (xAxisMode === 'multi-feature') {
      const featuresToTest = selectedXFeatureIdx === -1
        ? Array.from({ length: inputNodes }, (_, idx) => idx)
        : [selectedXFeatureIdx];

      for (const p of migratedPoints) {
        for (const f of featuresToTest) {
          const pX = p.features[f] ?? 0.5;
          const pY = p.label ?? 0;
          const d = Math.hypot(pX - sx_val, pY - sy_val);
          if (d < minD) {
            minD = d;
            nearest = { features: p.features, px: mx, py: my, hoverFeatureIdx: f, label: p.label };
          }
        }
      }
    } else {
      for (const p of migratedPoints) {
        const pX = p.features[0]; const pY = p.features[1] ?? 0;
        const d = Math.hypot(pX - px, pY - py);
        if (d < minD) {
          minD = d;
          nearest = { features: p.features, px: mx, py: my, label: p.label };
        }
      }
    }
    setHoverPt(nearest);

    if (!dragRef.current) return; 
    const dr = dragRef.current; 
    const dx = ((e.clientX - dr.sx) / dw) * (dr.vp.xMax - dr.vp.xMin); 
    const dy = ((e.clientY - dr.sy) / dh) * (dr.vp.yMax - dr.vp.yMin); 
    if (Math.abs(e.clientX - dr.sx) > 3 || Math.abs(e.clientY - dr.sy) > 3) {
      dr.moved = true;
    }
    vpRef.current = { 
      xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, 
      yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy 
    }; 
    setVpVer(v => v + 1); 
  }, [migratedPoints, xAxisMode, selectedXFeatureIdx, inputNodes]);

  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { 
    if (xAxisMode === 'multi-feature') {
      const ys = migratedPoints.map(p => p.label);
      const minVal = ys.length > 0 ? Math.min(...ys) : 0;
      const maxVal = ys.length > 0 ? Math.max(...ys) : 1;
      const padding = (maxVal - minVal) * 0.1 || 0.2;
      vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: minVal - padding, yMax: maxVal + padding };
    } else {
      vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    }
    setVpVer(v => v + 1); 
  }, [xAxisMode, migratedPoints]);
  const zoomBtn = useCallback((f: number) => { 
    const vp = vpRef.current; 
    const mx = (vp.xMin + vp.xMax) / 2; const my = (vp.yMin + vp.yMax) / 2; 
    vpRef.current = { 
      xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, 
      yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f 
    }; 
    setVpVer(v => v + 1); 
  }, []);

  // Training Loop
  useEffect(() => {
    if (!isTraining || migratedPoints.length === 0) return;
    
    let state = percState;
    if (trained) {
       state = { ...state, epoch: 0, lossHistory: [], converged: false };
    }
    
    const target = state.lossHistory.length + state.maxEpochs;
    setEpochTarget(target);
    
    const id = setInterval(() => {
      const steps = Math.max(1, Math.floor(state.maxEpochs / 50));
      for (let i = 0; i < steps; i++) {
        state = trainStep(migratedPoints, state);
        if (state.converged || state.epoch >= state.maxEpochs) break;
      }
      setPercState(state);
      
      if (state.converged || state.epoch >= state.maxEpochs) {
        clearInterval(id);
        setTrained(true);
        onTrainingComplete();
      }
    }, 40);
    
    return () => clearInterval(id);
  }, [isTraining, migratedPoints]); // eslint-disable-line
  
  useEffect(() => { pushMetrics(percState); }, [percState, pushMetrics]);

  // Renders
  useEffect(() => { 
    const c = dataRef.current; if (!c) return; 
    const r = () => drawDataCanvas(c, migratedPoints, percState, vpRef.current, dataset, inferResults, xAxisMode, selectedXFeatureIdx); 
    r(); 
    const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); 
    return () => ro.disconnect(); 
  }, [migratedPoints, percState, vpVer, inferResults, dataset, xAxisMode, selectedXFeatureIdx]);
  
  useEffect(() => { 
    const c = netRef.current; if (!c) return; 
    const ip = inferResults.length > 0 ? { features: inferResults[0].features, label: 0 } : null; 
    let iv = null;
    if (ip) {
       const res = predictPerceptron(ip, percState);
       iv = { hiddenActs: res.hiddenActs, pred: res.pred };
    }
    const r = () => drawNetworkCanvas(c, percState, ip, iv); 
    r(); 
    const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); 
    return () => ro.disconnect(); 
  }, [percState, inferResults]);

  // Loss Curve Render
  useEffect(() => {
    const canvas = lossRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
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
      ctx.globalAlpha = 0.6; ctx.fillText('LOSS CURVE (MSE)', 12, 18); ctx.globalAlpha = 1;

      const totalEp = epochTarget > 0 ? epochTarget : percState.maxEpochs;
      const padL = 40, padR = 16, padT = 30, padB = 24;
      
      if (percState.lossHistory.length < 1) {
        ctx.strokeStyle = border; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
        ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
        ctx.fillText('Train to see loss curve', w / 2, h / 2 + 10);
        ctx.globalAlpha = 1; ctx.textAlign = 'start';
        return;
      }

      const cw = w - padL - padR, ch = h - padT - padB;
      ctx.strokeStyle = border; ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

      const maxLoss = Math.max(...percState.lossHistory) * 1.1 || 1;

      ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
      for (let i = 0; i <= 4; i++) {
        const val = maxLoss * (1 - i / 4);
        const y = padT + (i / 4) * ch;
        ctx.fillText(val.toFixed(2), 2, y + 3);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.strokeStyle = border; ctx.lineWidth = 0.3; ctx.stroke();
      }
      for (let i = 0; i <= 4; i++) {
        const ep = Math.round((totalEp / 4) * i);
        const x = padL + (i / 4) * cw;
        ctx.fillText(String(ep), x - 6, h - 8);
      }
      ctx.globalAlpha = 1;

      ctx.beginPath();
      for (let i = 0; i < percState.lossHistory.length; i++) {
        const x = padL + (i / (totalEp - 1 || 1)) * cw;
        const y = padT + ((maxLoss - percState.lossHistory[i]) / maxLoss) * ch;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = primary; ctx.lineWidth = 1.5; ctx.stroke();

      const lastX = padL + ((percState.lossHistory.length - 1) / (totalEp - 1 || 1)) * cw;
      const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
      gradient.addColorStop(0, `${primary}20`); gradient.addColorStop(1, `${primary}02`);
      ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath();
      ctx.fillStyle = gradient; ctx.fill();
    };
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [percState.lossHistory, epochTarget, percState.maxEpochs]);

  const handleInfer = useCallback(() => {
    const rawFeatures = inferInputs.map(v => parseFloat(v));
    if (rawFeatures.some(isNaN)) return;

    const scaledFeatures = rawFeatures.map((v, i) => {
      if (dataset === 'import' && importStats && shouldScaleInputs) {
        const min = importStats.mins[i] ?? 0;
        const max = importStats.maxs[i] ?? 1;
        const range = max - min;
        return range === 0 ? 0.5 : (v - min) / range;
      }
      return v;
    });

    const { pred } = predictPerceptron({ features: scaledFeatures, label: 0 }, percState);
    setInferResults(prev => [{ features: scaledFeatures, pred, rawFeatures }, ...prev].slice(0, 5));
  }, [inferInputs, percState, dataset, importStats, shouldScaleInputs]);

  const handleManualHiddenWeight = (neuronIdx: number, weightIdx: number, val: number) => {
    setPercState(s => {
      const nextW = s.hiddenWeights.map(w => [...w]);
      nextW[neuronIdx][weightIdx] = val;
      return { ...s, hiddenWeights: nextW, converged: false };
    });
  };

  const handleManualHiddenBias = (neuronIdx: number, val: number) => {
    setPercState(s => {
      const nextB = [...s.hiddenBias];
      nextB[neuronIdx] = val;
      return { ...s, hiddenBias: nextB, converged: false };
    });
  };

  const handleManualOutWeight = (weightIdx: number, val: number) => {
    setPercState(s => {
      const nextW = [...s.outWeights];
      nextW[weightIdx] = val;
      return { ...s, outWeights: nextW, converged: false };
    });
  };

  const handleManualOutBias = (val: number) => {
    setPercState(s => ({ ...s, outBias: val, converged: false }));
  };

  const updateInferInput = (idx: number, val: string) => {
    setInferInputs(prev => {
      const next = [...prev];
      next[idx] = val;
      return next;
    });
  };

  // Data stats
  const stats = (() => {
    if (migratedPoints.length < 2) return null;
    let n = migratedPoints.length, class0 = 0, class1 = 0;
    for (const p of migratedPoints) {
      if (p.label === 0) class0++; else class1++;
    }
    return { n, class0, class1 };
  })();

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER MAP */}
      <div className="viz-scroll__section viz-scroll__section--canvas" ref={scatterContainerRef} style={{ position: 'relative' }}>
        {showDataPreview ? (
          <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--c-surface-container)', borderRadius: '12px', overflow: 'hidden' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--c-panel-border)', background: 'var(--c-surface-container-high)', fontSize: '11px', fontWeight: 600, color: 'var(--c-on-surface-variant)', display: 'flex', justifyContent: 'space-between' }}>
              <span>Data Preview ({migratedPoints.length} points)</span>
              <span>Showing first 50 rows</span>
            </div>
            <div style={{ flex: 1, overflow: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
                <thead style={{ position: 'sticky', top: 0, background: 'var(--c-surface-container-high)', zIndex: 1 }}>
                  <tr>
                    <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-on-surface-variant)' }}>#</th>
                    {Array.from({ length: inputNodes }).map((_, i) => (
                      <th key={i} style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-primary)' }}>x{i + 1}</th>
                    ))}
                    <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-tertiary)' }}>Label (y)</th>
                  </tr>
                </thead>
                <tbody>
                  {migratedPoints.slice(0, 50).map((pt, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                      <td style={{ padding: '6px 8px', color: 'var(--c-on-surface-variant)', opacity: 0.5 }}>{i + 1}</td>
                      {pt.features.map((f, fi) => (
                        <td key={fi} style={{ padding: '6px 8px', fontFamily: "'JetBrains Mono', monospace" }}>{f.toFixed(4)}</td>
                      ))}
                      <td style={{ padding: '6px 8px', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>{pt.label.toFixed(4)}</td>
                    </tr>
                  ))}
                  {migratedPoints.length === 0 && (
                    <tr><td colSpan={inputNodes + 2} style={{ padding: '24px', textAlign: 'center', opacity: 0.5 }}>No data available</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <canvas ref={dataRef} onContextMenu={e => e.preventDefault()} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={() => { handleMouseUp(); setHoverPt(null); }} onClick={dataset === 'custom' && xAxisMode === 'standard' ? handleDataClick : undefined} onWheel={handleWheel} style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' && xAxisMode === 'standard' ? 'crosshair' : 'grab' }} />
        )}
        <div className="viz-scatter-ctrls" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <button className="viz-scatter-btn" onClick={() => setShowDataPreview(p => !p)} title="Toggle Data Preview" style={{ background: showDataPreview ? 'var(--c-primary)' : undefined, color: showDataPreview ? '#fff' : undefined }}>
            <span className="material-symbols-rounded" style={{ fontSize: '18px' }}>table_chart</span>
          </button>
          {!showDataPreview && <button className="viz-scatter-btn" onClick={resetView} title="Reset view">⟲</button>}
        </div>
        
        {/* Dynamic X Axis Mapping & Multi-Feature Selector */}
        {!showDataPreview && inputNodes > 1 && (
          <div style={{
            position: 'absolute',
            bottom: '12px',
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'var(--c-surface-container-high)',
            border: '1px solid var(--c-panel-border)',
            borderRadius: '24px',
            padding: '4px 12px',
            boxShadow: '0 4px 20px rgba(0,0,0,0.4)',
            zIndex: 10
          }}>
            <span style={{ fontSize: '10px', fontWeight: 'bold', color: 'var(--c-on-surface-variant)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>X Axis:</span>
            <button
              onClick={() => {
                setXAxisMode('standard');
                vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
                setVpVer(v => v + 1);
              }}
              style={{
                background: xAxisMode === 'standard' ? 'var(--c-primary)' : 'transparent',
                color: xAxisMode === 'standard' ? '#fff' : 'var(--c-on-surface-variant)',
                border: 'none',
                borderRadius: '16px',
                padding: '4px 10px',
                fontSize: '10px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              Standard 2D
            </button>
            <button
              onClick={() => {
                setXAxisMode('multi-feature');
                if (selectedXFeatureIdx === 0 && inputNodes > 2) {
                  setSelectedXFeatureIdx(-1);
                }
                const ys = migratedPoints.map(p => p.label);
                const minVal = ys.length > 0 ? Math.min(...ys) : 0;
                const maxVal = ys.length > 0 ? Math.max(...ys) : 1;
                const padding = (maxVal - minVal) * 0.1 || 0.2;
                vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: minVal - padding, yMax: maxVal + padding };
                setVpVer(v => v + 1);
              }}
              style={{
                background: xAxisMode === 'multi-feature' ? 'var(--c-primary)' : 'transparent',
                color: xAxisMode === 'multi-feature' ? '#fff' : 'var(--c-on-surface-variant)',
                border: 'none',
                borderRadius: '16px',
                padding: '4px 10px',
                fontSize: '10px',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease'
              }}
            >
              Feature vs Output
            </button>
            
            {xAxisMode === 'multi-feature' && (
              <select
                value={selectedXFeatureIdx}
                onChange={e => setSelectedXFeatureIdx(Number(e.target.value))}
                style={{
                  background: 'var(--c-surface-container-highest)',
                  color: 'var(--c-on-surface)',
                  border: '1px solid var(--c-panel-border)',
                  borderRadius: '12px',
                  padding: '2px 8px',
                  fontSize: '10px',
                  fontWeight: 600,
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                <option value={-1}>All Features</option>
                {Array.from({ length: inputNodes }).map((_, idx) => (
                  <option key={idx} value={idx}>Feature x{idx+1}</option>
                ))}
              </select>
            )}
          </div>
        )}

        {/* Hover Tooltip Popup */}
        {hoverPt && (
          <div className="viz-tooltip" style={{ 
            position: 'absolute',
            left: hoverPt.px + 12, 
            top: hoverPt.py - 8,
            background: 'var(--c-surface-container-high)',
            border: '1px solid var(--c-panel-border)',
            borderRadius: '6px',
            padding: '8px 12px',
            fontSize: '11px',
            boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
            color: 'var(--c-on-surface)',
            pointerEvents: 'none',
            zIndex: 99
          }}>
            {hoverPt.hoverFeatureIdx !== undefined ? (
              <div>
                <div style={{ fontWeight: 'bold', color: FEATURE_COLORS[hoverPt.hoverFeatureIdx % FEATURE_COLORS.length], marginBottom: '4px' }}>
                  Feature x{hoverPt.hoverFeatureIdx + 1}
                </div>
                <div>Value: <span style={{ fontFamily: 'monospace' }}>{hoverPt.features[hoverPt.hoverFeatureIdx]?.toFixed(4) ?? '0.0000'}</span></div>
                <div>Label/Output: <span style={{ fontFamily: 'monospace' }}>{hoverPt.label?.toFixed(4) ?? '0.0000'}</span></div>
              </div>
            ) : (
              <div>
                <strong style={{ color: 'var(--c-primary)', display: 'block', marginBottom: '4px' }}>Point details:</strong>
                {hoverPt.features.map((v, i) => (
                  <div key={i}>x{i+1}: <span style={{ fontFamily: 'monospace' }}>{v?.toFixed(3) ?? '0.000'}</span></div>
                ))}
                {hoverPt.label !== undefined && (
                  <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', marginTop: '4px', paddingTop: '4px' }}>
                    Label: <span style={{ fontFamily: 'monospace' }}>{hoverPt.label}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        <div className="viz-scatter-ctrls">
          <button className="viz-scatter-btn" onClick={resetView} title="Reset view">⟲</button>
          <button className="viz-scatter-btn" onClick={() => zoomBtn(0.8)} title="Zoom In">+</button>
          <button className="viz-scatter-btn" onClick={() => zoomBtn(1.2)} title="Zoom Out">−</button>
          <button className="viz-scatter-btn" onClick={() => {
            if (document.fullscreenElement) document.exitFullscreen();
            else scatterContainerRef.current?.requestFullscreen();
          }} title="Full Screen">⛶</button>
        </div>
      </div>

      {/* 2. LOSS CURVE */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={lossRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. NEURAL NETWORK DIAGRAM */}
      <div className="viz-scroll__section" style={{ minHeight: `${Math.max(220, Math.max(inputNodes, numPerceptrons) * 60 + 60)}px`, position: 'relative' }}>
        <canvas ref={netRef} style={{ width: '100%', height: '100%', display: 'block', borderRadius: '8px', background: 'var(--c-surface-container-low)' }} />
      </div>

      {/* EVALUATION MATRIX */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        {outAct === 'linear' ? (
          // REGRESSION PERFORMANCE MATRIX
          <>
            <div className="viz-ctrl__header">
              <span className="viz-ctrl__title">REGRESSION PERFORMANCE MATRIX</span>
              <span className="viz-ctrl__subtitle">Evaluation metrics for model fit</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', textAlign: 'center', fontSize: '11px', marginTop: '10px' }}>
              <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.regressionMetrics.r2.toFixed(4)}</div>
                <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>R² (Accuracy of Fit)</div>
                <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Proportion of variance explained</div>
              </div>
              <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.regressionMetrics.mse.toFixed(4)}</div>
                <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>MSE (Mean Squared Error)</div>
                <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Average squared prediction error</div>
              </div>
              <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.regressionMetrics.mae.toFixed(4)}</div>
                <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>MAE (Mean Absolute Error)</div>
                <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Average magnitude of errors</div>
              </div>
              <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '18px' }}>{metrics.regressionMetrics.rmse.toFixed(4)}</div>
                <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>RMSE (Root MSE)</div>
                <div style={{ fontSize: '9px', color: 'var(--c-on-surface-variant)', opacity: 0.7 }}>Standard deviation of residuals</div>
              </div>
            </div>
          </>
        ) : (
          // BINARY CONFUSION MATRIX
          <>
            <div className="viz-ctrl__header">
              <span className="viz-ctrl__title">CONFUSION MATRIX</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: '4px', textAlign: 'center', fontSize: '11px', marginTop: '10px' }}>
              <div></div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 0</div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 1</div>
              <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 0</div>
              <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '14px' }}>{metrics.binaryMetrics.tn}</div><div>TN</div>
              </div>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{metrics.binaryMetrics.fp}</div><div>FP</div>
              </div>
              <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 1</div>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{metrics.binaryMetrics.fn}</div><div>FN</div>
              </div>
              <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '14px' }}>{metrics.binaryMetrics.tp}</div><div>TP</div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* 4. MANUAL WEIGHT CONTROL */}
      <div className="viz-scroll__section viz-scroll__section--controls">
        <div className="viz-ctrl__header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span className="viz-ctrl__title">MANUAL WEIGHTS</span>
            <span className="viz-ctrl__subtitle">Adjust neural network parameters</span>
          </div>
          <button className="viz-scatter-btn" style={{ padding: '4px 8px', width: 'auto', fontSize: '11px' }} onClick={() => setAdvancedMode(m => !m)}>
            {advancedMode ? 'Basic Mode' : 'Advanced Mode'}
          </button>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', borderBottom: '1px solid var(--c-panel-border)', paddingBottom: '8px' }}>
          <button 
             onClick={() => setSelectedLayer('hidden')}
             style={{ padding: '6px 12px', background: selectedLayer === 'hidden' ? 'var(--c-surface-variant)' : 'transparent', color: selectedLayer === 'hidden' ? 'var(--c-on-surface)' : 'var(--c-on-surface-variant)', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: selectedLayer === 'hidden' ? 'bold' : 'normal' }}>
             Hidden Layer
          </button>
          <button 
             onClick={() => setSelectedLayer('output')}
             style={{ padding: '6px 12px', background: selectedLayer === 'output' ? 'var(--c-surface-variant)' : 'transparent', color: selectedLayer === 'output' ? 'var(--c-on-surface)' : 'var(--c-on-surface-variant)', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: selectedLayer === 'output' ? 'bold' : 'normal' }}>
             Output Layer
          </button>
        </div>

        {selectedLayer === 'hidden' && (
          <>
            <div style={{ marginBottom: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
               {Array.from({ length: numPerceptrons }).map((_, i) => (
                 <button key={i} onClick={() => setSelectedNeuron(i)}
                   style={{ padding: '4px 8px', fontSize: '12px', background: selectedNeuron === i ? 'var(--c-primary)' : 'var(--c-surface-container-highest)', color: selectedNeuron === i ? '#fff' : 'var(--c-on-surface)', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                   Neuron {i+1}
                 </button>
               ))}
            </div>
            
            <div className="viz-ctrl__sliders" style={{ padding: '12px 0' }}>
              {percState.hiddenWeights[selectedNeuron]?.map((w, i) => (
                <div key={`hw${i}`} style={{ marginBottom: '12px' }}>
                  <div className="viz-ctrl__slider-row">
                    <label>Weight (w{i+1})</label>
                    <span className="viz-ctrl__slider-val">{w?.toFixed(3) ?? '0.000'}</span>
                  </div>
                  <input type="range" className="control__range" min="-5" max="5" step="0.01" value={w} onChange={e => handleManualHiddenWeight(selectedNeuron, i, Number(e.target.value))} />
                  <div className="control__range-labels"><span>-5</span><span>5</span></div>
                </div>
              ))}
              <div style={{ marginBottom: '12px' }}>
                <div className="viz-ctrl__slider-row">
                  <label>Bias (b)</label>
                  <span className="viz-ctrl__slider-val">{percState.hiddenBias[selectedNeuron]?.toFixed(3) ?? '0.000'}</span>
                </div>
                <input type="range" className="control__range" min="-5" max="5" step="0.01" value={percState.hiddenBias[selectedNeuron] || 0} onChange={e => handleManualHiddenBias(selectedNeuron, Number(e.target.value))} />
                <div className="control__range-labels"><span>-5</span><span>5</span></div>
              </div>
            </div>
          </>
        )}

        {selectedLayer === 'output' && (
          <div className="viz-ctrl__sliders" style={{ padding: '12px 0' }}>
            {percState.outWeights.map((w, i) => (
              <div key={`ow${i}`} style={{ marginBottom: '12px' }}>
                <div className="viz-ctrl__slider-row">
                  <label>Weight from h{i+1}</label>
                  <span className="viz-ctrl__slider-val">{w?.toFixed(3) ?? '0.000'}</span>
                </div>
                <input type="range" className="control__range" min="-5" max="5" step="0.01" value={w} onChange={e => handleManualOutWeight(i, Number(e.target.value))} />
                <div className="control__range-labels"><span>-5</span><span>5</span></div>
              </div>
            ))}
            <div style={{ marginBottom: '12px' }}>
              <div className="viz-ctrl__slider-row">
                <label>Output Bias (b)</label>
                <span className="viz-ctrl__slider-val">{percState.outBias?.toFixed(3) ?? '0.000'}</span>
              </div>
              <input type="range" className="control__range" min="-5" max="5" step="0.01" value={percState.outBias} onChange={e => handleManualOutBias(Number(e.target.value))} />
              <div className="control__range-labels"><span>-5</span><span>5</span></div>
            </div>
          </div>
        )}
        
        {advancedMode && (
          <div style={{ padding: '12px', background: 'var(--c-surface-container)', borderRadius: '6px', fontSize: '12px', color: 'var(--c-on-surface-variant)', marginTop: '16px' }}>
            <strong>Advanced Mode Info:</strong><br/>
            Network Architecture: MLP with 1 hidden layer.<br/>
            Hidden activation: {activation}. Output activation: sigmoid.<br/>
            Training uses backpropagation. You can edit every single weight matrix manually via the tabs above to explore its effect on the decision boundary.
          </div>
        )}
      </div>

      {/* PERCEPTRON ALGORITHM TRACKER */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">BACKPROPAGATION TRACKER</span>
          <span className="viz-ctrl__subtitle">Mathematical breakdown of network learning</span>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '12px' }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', color: 'var(--c-on-surface-variant)', background: 'rgba(0,0,0,0.2)', padding: '10px', overflowX: 'auto', whiteSpace: 'nowrap' }}>
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Forward Pass (Prediction)</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Hidden Layer: <span style={{ color: '#a855f7' }}>h = σ(W_h · x + b_h)</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Output Layer: <span style={{ color: 'var(--c-primary)' }}>ŷ = {outAct === 'linear' ? 'W_o · h + b_o' : 'σ(W_o · h + b_o)'}</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>2. Loss Function (MSE)</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Loss: <span style={{ color: 'var(--c-error)' }}>L = ½(ŷ - y)²</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>3. Backward Pass (Gradients)</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Output Error: <span style={{ color: 'var(--c-error)' }}>δ_o = (ŷ - y) · {outAct === 'linear' ? '1' : 'σ\'(z_o)'}</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Hidden Error: <span style={{ color: 'var(--c-tertiary)' }}>δ_h = (W_oᵀ · δ_o) ⊙ σ\'(z_h)</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>4. Gradient Descent Update</div>
            <div style={{ color: 'var(--c-on-surface)' }}>W_o := <span style={{ color: 'var(--c-primary)' }}>W_o - η · (δ_o · hᵀ)</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>W_h := <span style={{ color: '#a855f7' }}>W_h - η · (δ_h · xᵀ)</span></div>
          </div>
        </div>
      </div>

      {/* 5. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">INFERENCE</span>
          <span className="viz-ctrl__subtitle">Forward pass with custom inputs</span>
        </div>

        {dataset === 'import' && importStats && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)' }}>
            <input
              type="checkbox"
              id="perc-scale-inputs"
              checked={shouldScaleInputs}
              onChange={e => setShouldScaleInputs(e.target.checked)}
              style={{ cursor: 'pointer', accentColor: 'var(--c-primary)' }}
            />
            <label htmlFor="perc-scale-inputs" style={{ fontSize: '11px', fontWeight: 500, color: 'var(--c-on-surface-variant)', cursor: 'pointer', userSelect: 'none' }}>
              Scale manual inputs using dataset limits (Min-Max)
            </label>
          </div>
        )}

        <div className="viz-infer__input-row" style={{ flexWrap: 'wrap' }}>
          {inferInputs.map((val, i) => (
             <div className="viz-infer__field" key={`in${i}`}>
               <label>x{i+1}</label>
               <input className="viz-infer__input" type="number" step="0.01" value={val} onChange={e => updateInferInput(i, e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '64px' }} />
             </div>
          ))}
          <button className="viz-infer__btn" onClick={handleInfer} style={{ marginTop: 'auto', marginBottom: 'auto' }}>Predict y</button>
          <div className="viz-infer__result" style={{ marginTop: 'auto', marginBottom: 'auto' }}>
            <label>ŷ</label>
            <span className="viz-infer__y">{inferResults.length > 0 ? inferResults[0].pred.toFixed(3) : '—'}</span>
          </div>
        </div>
        
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header">
              <span>Inputs x</span><span>Output ŷ</span>
            </div>
            {inferResults.map((r, i) => {
              const displayFeatures = r.rawFeatures || r.features;
              return (
                <div key={i} className="viz-infer__history-row">
                  <span>[{displayFeatures.map(f => f.toFixed(2)).join(', ')}]</span>
                  <span style={{ color: r.pred >= 0.5 ? 'var(--c-primary)' : 'var(--c-tertiary)', fontWeight: 'bold' }}>{r.pred.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* 6. DATA STATS */}
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">DATA STATISTICS</span>
          </div>
          <div className="viz-stats__grid">
            <div className="viz-stats__item"><span className="viz-stats__label">N</span><span className="viz-stats__val">{stats.n}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Features</span><span className="viz-stats__val">{inputNodes}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Hidden Nodes</span><span className="viz-stats__val" style={{ color: 'var(--c-tertiary)' }}>{numPerceptrons}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 0</span><span className="viz-stats__val" style={{ color: '#a855f7' }}>{stats.class0}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 1</span><span className="viz-stats__val" style={{ color: '#4ade80' }}>{stats.class1}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
