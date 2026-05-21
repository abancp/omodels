import { usePersistentState } from '../../hooks/usePersistentState';
import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateData, trainStep, predictMLP, type Point, type MLPState, type ActivationType, type MLPLayer } from './math';
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

const ACT_FORMULAS: Record<string, { fn: string; deriv: string }> = {
  sigmoid: { fn: "σ(x) = 1 / (1 + e⁻ˣ)", deriv: "σ'(x) = σ(x) · (1 - σ(x))" },
  relu: { fn: "ReLU(x) = max(0, x)", deriv: "ReLU'(x) = 1 if x > 0 else 0" },
  tanh: { fn: "tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)", deriv: "tanh'(x) = 1 - tanh²(x)" },
  step: { fn: "step(x) = 1 if x ≥ 0 else 0", deriv: "step'(x) = 0" },
  linear: { fn: "f(x) = x", deriv: "f'(x) = 1" },
  softmax: { fn: "S(x_i) = e^(x_i) / Σ e^(x_j)", deriv: "S'(x) ≈ S(x)(1 - S(x))" },
};

export default function MLPVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const netRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const scatterContainerRef = useRef<HTMLDivElement>(null);

  const lr = (params.learningRate as number) ?? 0.01;
  const maxEp = (params.maxEpochs as number) ?? 500;
  const inputNodes = (params.inputNodes as number) ?? 2;
  const numLayers = (params.numLayers as number) ?? 2;

  const numPoints = (datasetParams.points as number) ?? 200;
  const noise = (datasetParams.noise as number) ?? 0.1;

  const getLayerConfig = useCallback(() => {
    const layers: MLPLayer[] = [];
    let prevNodes = inputNodes;
    for (let i = 1; i <= numLayers; i++) {
      const nodes = (params[`l${i}Nodes`] as number) ?? 4;
      const act = (params[`l${i}Act`] as ActivationType) ?? 'relu';
      layers.push({
        nodes,
        activation: act,
        weights: Array.from({ length: nodes }, () => Array.from({ length: prevNodes }, () => (Math.random() - 0.5) * Math.sqrt(2 / prevNodes))), // He init
        biases: Array.from({ length: nodes }, () => 0.01)
      });
      prevNodes = nodes;
    }
    // Output Layer
    const outNodes = (params.outNodes as number) ?? 1;
    const outAct = (params.outAct as ActivationType) ?? 'sigmoid';
    layers.push({
      nodes: outNodes,
      activation: outAct,
      weights: Array.from({ length: outNodes }, () => Array.from({ length: prevNodes }, () => (Math.random() - 0.5) * Math.sqrt(2 / prevNodes))),
      biases: Array.from({ length: outNodes }, () => 0)
    });
    return layers;
  }, [params, inputNodes, numLayers]);

  const [points, setPoints] = usePersistentState<Point[]>('omodels-mlp-points', []);
  const [mlpState, setMlpState] = usePersistentState<MLPState>('omodels-mlp-mlpState', () => ({
    numInputs: inputNodes,
    layers: getLayerConfig(),
    learningRate: lr,
    epoch: 0,
    maxEpochs: maxEp,
    lossHistory: [],
    converged: false
  }));
  
  const [epochTarget, setEpochTarget] = usePersistentState('omodels-mlp-epochTarget', 0);
  const [trained, setTrained] = usePersistentState('omodels-mlp-trained', false);
  const [shouldScaleInputs, setShouldScaleInputs] = usePersistentState('omodels-mlp-shouldScaleInputs', true);
  const [xAxisMode, setXAxisMode] = usePersistentState<'standard' | 'multi-feature'>('omodels-mlp-xAxisMode', 'standard');
  const [selectedXFeatureIdx, setSelectedXFeatureIdx] = usePersistentState<number>('omodels-mlp-selectedXFeatureIdx', 0);
  const [showDataPreview, setShowDataPreview] = usePersistentState('omodels-mlp-showDataPreview', false);
  const [slowMode, setSlowMode] = useState(false);
  const [hoverPt, setHoverPt] = useState<{ features: number[]; px: number; py: number; hoverFeatureIdx?: number; label: number } | null>(null);

  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current; moved?: boolean } | null>(null);
  
  // Advanced Mode State
  const [selectedLayerIdx, setSelectedLayerIdx] = useState<number>(0);
  const [selectedNeuronIdx, setSelectedNeuronIdx] = useState<number>(0);

  // Inference
  const [inferInputs, setInferInputs] = usePersistentState<string[]>('omodels-mlp-inferInputs', Array(inputNodes).fill('0.50'));
  const [inferResults, setInferResults] = usePersistentState<{ features: number[]; pred: number[]; rawFeatures?: number[] }[]>('omodels-mlp-inferResults', []);

  const outAct = (params.outAct as ActivationType) ?? 'sigmoid';
  const outNodes = (params.outNodes as number) ?? 1;

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
    let multiclassMetrics = { tp_c: new Array(outNodes).fill(0), fp_c: new Array(outNodes).fill(0), fn_c: new Array(outNodes).fill(0), acc: 0, macroPrec: 0, macroRec: 0, macroF1: 0 };

    if (total === 0) {
      return { regressionMetrics, binaryMetrics, multiclassMetrics };
    }

    if (outAct === 'linear') {
      const yMean = migratedPoints.reduce((acc, p) => acc + p.label, 0) / total;
      let ssTot = 0, ssRes = 0, mse = 0, mae = 0;
      for (const p of migratedPoints) {
        const { pred } = predictMLP(p.features, mlpState);
        const yHat = pred[0] ?? 0;
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
    } else if (outNodes > 1) {
      let correct = 0;
      const tp_c = new Array(outNodes).fill(0);
      const fp_c = new Array(outNodes).fill(0);
      const fn_c = new Array(outNodes).fill(0);

      for (const p of migratedPoints) {
        const { pred } = predictMLP(p.features, mlpState);
        const predClass = pred.indexOf(Math.max(...pred));
        const trueClass = p.label;
        if (predClass === trueClass) {
          correct++;
          if (trueClass < outNodes) tp_c[trueClass]++;
        } else {
          if (trueClass < outNodes) fn_c[trueClass]++;
          if (predClass < outNodes) fp_c[predClass]++;
        }
      }
      const acc = correct / total;
      let precSum = 0, recSum = 0;
      for (let c = 0; c < outNodes; c++) {
        const prec = (tp_c[c] + fp_c[c]) > 0 ? tp_c[c] / (tp_c[c] + fp_c[c]) : 0;
        const rec = (tp_c[c] + fn_c[c]) > 0 ? tp_c[c] / (tp_c[c] + fn_c[c]) : 0;
        precSum += prec;
        recSum += rec;
      }
      const macroPrec = precSum / outNodes;
      const macroRec = recSum / outNodes;
      const macroF1 = (macroPrec + macroRec) > 0 ? 2 * macroPrec * macroRec / (macroPrec + macroRec) : 0;
      multiclassMetrics = { tp_c, fp_c, fn_c, acc, macroPrec, macroRec, macroF1 };
    } else {
      let tp = 0, tn = 0, fp = 0, fn = 0;
      for (const p of migratedPoints) {
        const { pred } = predictMLP(p.features, mlpState);
        const predClass = pred[0] >= 0.5 ? 1 : 0;
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

    return { regressionMetrics, binaryMetrics, multiclassMetrics };
  }, [migratedPoints, mlpState, outAct, outNodes]);

  const pushMetrics = useCallback((s: MLPState) => {
    const loss = s.lossHistory[s.lossHistory.length - 1] ?? 0;

    if (outAct === 'linear') {
      onMetricsUpdate([
        { label: 'Loss (MSE)', value: loss.toFixed(4), isPrimary: true },
        { label: 'R²', value: metrics.regressionMetrics.r2.toFixed(4) },
        { label: 'MAE', value: metrics.regressionMetrics.mae.toFixed(4) },
        { label: 'Epochs', value: String(s.epoch) },
        { label: 'Converged', value: s.converged ? 'Yes' : 'No' },
      ]);
    } else if (outNodes > 1) {
      onMetricsUpdate([
        { label: 'Accuracy', value: (metrics.multiclassMetrics.acc * 100).toFixed(1) + '%', isPrimary: true },
        { label: 'Loss (MSE)', value: loss.toFixed(4) },
        { label: 'Macro F1', value: metrics.multiclassMetrics.macroF1.toFixed(3) },
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
  }, [outAct, outNodes, metrics, onMetricsUpdate]);

  const resetState = useCallback(() => {
    setMlpState(s => ({
      ...s,
      numInputs: inputNodes,
      layers: getLayerConfig(),
      epoch: 0,
      lossHistory: [],
      converged: false
    }));
    setEpochTarget(0);
    setTrained(false);
    setInferResults([]);
  }, [inputNodes, getLayerConfig]);

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
        const { pred } = predictMLP(feats, mlpState);
        const yHat = pred[0] ?? 0;
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
    } else if (outNodes > 1) {
      let correct = 0;
      const numClasses = outNodes;
      const confusionMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
      for (const p of testData) {
        const feats = p.features || [p.x ?? 0, p.y ?? 0];
        const { pred } = predictMLP(feats, mlpState);
        const predClass = pred.indexOf(Math.max(...pred));
        const trueClass = p.label ?? 0;
        if (predClass === trueClass) correct++;
        if (trueClass < numClasses && predClass < numClasses) confusionMatrix[trueClass][predClass]++;
        results.predictions.push({ features: feats, actual: trueClass, predicted: predClass });
      }
      const acc = correct / total;
      let precSum = 0, recSum = 0;
      const perClass: any[] = [];
      for (let c = 0; c < numClasses; c++) {
        const tp = confusionMatrix[c][c];
        const fp = confusionMatrix.reduce((s, row, r) => s + (r !== c ? row[c] : 0), 0);
        const fn = confusionMatrix[c].reduce((s, v, ci) => s + (ci !== c ? v : 0), 0);
        const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
        precSum += prec; recSum += rec;
        perClass.push({ class: c, tp, fp, fn, precision: prec, recall: rec, f1, support: tp + fn });
      }
      results.type = 'multiclass';
      results.accuracy = acc;
      results.macroPrecision = precSum / numClasses;
      results.macroRecall = recSum / numClasses;
      results.macroF1 = (results.macroPrecision + results.macroRecall) > 0 ? 2 * results.macroPrecision * results.macroRecall / (results.macroPrecision + results.macroRecall) : 0;
      results.confusionMatrix = confusionMatrix;
      results.perClass = perClass;
    } else {
      let tp = 0, tn = 0, fp = 0, fn = 0;
      for (const p of testData) {
        const feats = p.features || [p.x ?? 0, p.y ?? 0];
        const { pred } = predictMLP(feats, mlpState);
        const predClass = pred[0] >= 0.5 ? 1 : 0;
        const trueClass = p.label ?? 0;
        if (trueClass === 1 && predClass === 1) tp++;
        else if (trueClass === 0 && predClass === 0) tn++;
        else if (trueClass === 0 && predClass === 1) fp++;
        else fn++;
        results.predictions.push({ features: feats, actual: trueClass, predicted: predClass, confidence: pred[0] });
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
    // Convert {features,label} to MLP's {features,label} format, preferring pre-mapped high-dimensional features
    const pts = importedData.map((p: any) => ({
      features: p.features || [p.x, ...(p.y !== undefined ? [p.y] : [])],
      label: p.cls ?? p.label ?? 0,
    }));
    setPoints(pts);
    resetState();
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
    
    setMlpState(s => {
      if (s.numInputs === inputNodes && s.layers.length === numLayers + 1) return s;
      setEpochTarget(0);
      setTrained(false);
      return {
        ...s,
        numInputs: inputNodes,
        layers: getLayerConfig(),
        epoch: 0,
        lossHistory: [],
        converged: false
      };
    });
  }, [inputNodes, numLayers, getLayerConfig]);

  useEffect(() => { 
    if (dataset === 'custom' || dataset === 'import') {
      if (dataset === 'custom') { setPoints([]); resetState(); }
      return; 
    }
    setPoints(generateData(dataset, numPoints, noise, inputNodes)); 
    resetState();
  }, [dataset, numPoints, noise, inputNodes, numLayers, resetState]);

  useEffect(() => { 
    if (resetVersion === 0) return; 
    resetState();
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; 
    setVpVer(v => v + 1); 
  }, [resetVersion, resetState]);

  useEffect(() => { 
    setMlpState(s => {
      const nextLayers = s.layers.map((layer, idx) => {
         if (idx < numLayers) {
             const act = params[`l${idx+1}Act`] as ActivationType;
             if (act && act !== layer.activation) return { ...layer, activation: act };
         } else if (idx === s.layers.length - 1) {
             const outAct = params.outAct as ActivationType;
             if (outAct && outAct !== layer.activation) return { ...layer, activation: outAct };
         }
         return layer;
      });
      return { ...s, layers: nextLayers, learningRate: lr, maxEpochs: maxEp };
    }); 
  }, [params, lr, maxEp, numLayers]);

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

  // Training Loop
  useEffect(() => {
    if (!isTraining || migratedPoints.length === 0) return;
    
    let state = mlpState;
    if (trained) {
       state = { ...state, epoch: 0, lossHistory: [], converged: false };
    }
    
    const target = state.lossHistory.length + state.maxEpochs;
    setEpochTarget(target);
    
    const id = setInterval(() => {
      const steps = slowMode ? 1 : Math.max(1, Math.floor(state.maxEpochs / 50));
      for (let i = 0; i < steps; i++) {
        state = trainStep(migratedPoints, state);
        if (state.converged || state.epoch >= state.maxEpochs) break;
      }
      setMlpState(state);
      
      if (state.converged || state.epoch >= state.maxEpochs) {
        clearInterval(id);
        setTrained(true);
        onTrainingComplete();
      }
    }, slowMode ? 200 : 40);
    
    return () => clearInterval(id);
  }, [isTraining, migratedPoints, slowMode]); // eslint-disable-line
  
  useEffect(() => { pushMetrics(mlpState); }, [mlpState, pushMetrics]);

  // Renders
  useEffect(() => { 
    const c = dataRef.current; if (!c) return; 
    const r = () => drawDataCanvas(c, migratedPoints, mlpState, vpRef.current, dataset, inferResults, xAxisMode, selectedXFeatureIdx); 
    r(); 
    const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); 
    return () => ro.disconnect(); 
  }, [migratedPoints, mlpState, vpVer, inferResults, dataset, xAxisMode, selectedXFeatureIdx]);
  
  useEffect(() => { 
    const c = netRef.current; if (!c) return; 
    const ip = inferResults.length > 0 ? { features: inferResults[0].features, label: 0 } : null; 
    let ivActs = null;
    let ivPred = null;
    if (ip) {
       const res = predictMLP(ip.features, mlpState);
       ivActs = res.acts;
       ivPred = res.pred;
    }
    const r = () => drawNetworkCanvas(c, mlpState, ip, ivActs, ivPred); 
    r(); 
    const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); 
    return () => ro.disconnect(); 
  }, [mlpState, inferResults]);

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

      const totalEp = epochTarget > 0 ? epochTarget : mlpState.maxEpochs;
      const padL = 40, padR = 16, padT = 30, padB = 24;
      
      if (mlpState.lossHistory.length < 1) {
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

      const maxLoss = Math.max(...mlpState.lossHistory) * 1.1 || 1;

      ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
      for (let i = 0; i <= 4; i++) {
        const val = maxLoss * (1 - i / 4);
        const y = padT + (i / 4) * ch;
        ctx.fillText(val.toFixed(2), 2, y + 3);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.strokeStyle = border; ctx.lineWidth = 0.3; ctx.stroke();
      }
      ctx.globalAlpha = 1;

      ctx.beginPath();
      for (let i = 0; i < mlpState.lossHistory.length; i++) {
        const x = padL + (i / (totalEp - 1 || 1)) * cw;
        const y = padT + ((maxLoss - mlpState.lossHistory[i]) / maxLoss) * ch;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = primary; ctx.lineWidth = 1.5; ctx.stroke();

      const lastX = padL + ((mlpState.lossHistory.length - 1) / (totalEp - 1 || 1)) * cw;
      const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
      gradient.addColorStop(0, `${primary}20`); gradient.addColorStop(1, `${primary}02`);
      ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath();
      ctx.fillStyle = gradient; ctx.fill();
    };
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [mlpState.lossHistory, epochTarget, mlpState.maxEpochs]);

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

    const { pred } = predictMLP(scaledFeatures, mlpState);
    setInferResults(prev => [{ features: scaledFeatures, pred, rawFeatures }, ...prev].slice(0, 5));
  }, [inferInputs, mlpState, dataset, importStats, shouldScaleInputs]);

  const updateInferInput = (idx: number, val: string) => {
    setInferInputs(prev => {
      const next = [...prev];
      next[idx] = val;
      return next;
    });
  };

  const handleManualWeight = (layerIdx: number, neuronIdx: number, weightIdx: number, val: number) => {
    setMlpState(s => {
      const nextLayers = s.layers.map((l, i) => {
        if (i !== layerIdx) return l;
        const nextW = l.weights.map((w, ni) => ni === neuronIdx ? w.map((cw, wi) => wi === weightIdx ? val : cw) : w);
        return { ...l, weights: nextW };
      });
      return { ...s, layers: nextLayers, converged: false };
    });
  };

  const handleManualBias = (layerIdx: number, neuronIdx: number, val: number) => {
    setMlpState(s => {
      const nextLayers = s.layers.map((l, i) => {
        if (i !== layerIdx) return l;
        const nextB = l.biases.map((cb, ni) => ni === neuronIdx ? val : cb);
        return { ...l, biases: nextB };
      });
      return { ...s, layers: nextLayers, converged: false };
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

  const currentLayer = mlpState.layers[selectedLayerIdx];

  return (
    <div className="viz-scroll">
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
      </div>

      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={lossRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      <div className="viz-scroll__section" style={{ minHeight: '300px', position: 'relative' }}>
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
        ) : outNodes > 1 ? (
          // MULTI-CLASS PERFORMANCE MATRIX
          <>
            <div className="viz-ctrl__header">
              <span className="viz-ctrl__title">CLASS-WISE METRICS</span>
              <span className="viz-ctrl__subtitle">Evaluation metrics per target class</span>
            </div>
            
            <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', marginBottom: '12px', marginTop: '12px' }}>
              <div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '18px' }}>{(metrics.multiclassMetrics.acc * 100).toFixed(2)}%</div>
              <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>Overall Accuracy</div>
            </div>

            <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse', marginTop: '8px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-on-surface-variant)', textAlign: 'left' }}>
                  <th style={{ padding: '6px 4px' }}>Class</th>
                  <th style={{ padding: '6px 4px' }}>Precision</th>
                  <th style={{ padding: '6px 4px' }}>Recall</th>
                  <th style={{ padding: '6px 4px' }}>F1-Score</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: outNodes }).map((_, c) => {
                  const tp_c = metrics.multiclassMetrics.tp_c[c] ?? 0;
                  const fp_c = metrics.multiclassMetrics.fp_c[c] ?? 0;
                  const fn_c = metrics.multiclassMetrics.fn_c[c] ?? 0;
                  const prec = (tp_c + fp_c) > 0 ? tp_c / (tp_c + fp_c) : 0;
                  const rec = (tp_c + fn_c) > 0 ? tp_c / (tp_c + fn_c) : 0;
                  const f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
                  return (
                    <tr key={c} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '6px 4px', fontWeight: 'bold' }}>Class {c}</td>
                      <td style={{ padding: '6px 4px' }}>{prec.toFixed(3)}</td>
                      <td style={{ padding: '6px 4px' }}>{rec.toFixed(3)}</td>
                      <td style={{ padding: '6px 4px' }}>{f1.toFixed(3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        ) : (
          // BINARY CONFUSION MATRIX
          <>
            <div className="viz-ctrl__header">
              <span className="viz-ctrl__title">CLASSIFICATION METRICS</span>
            </div>
            
            <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', marginBottom: '16px', marginTop: '12px' }}>
              <div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '18px' }}>{(metrics.binaryMetrics.acc * 100).toFixed(2)}%</div>
              <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>Overall Accuracy</div>
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

      <div className="viz-scroll__section viz-scroll__section--controls">
        <div className="viz-ctrl__header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span className="viz-ctrl__title">MANUAL WEIGHTS</span>
            <span className="viz-ctrl__subtitle">Inspect deep network weights</span>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', borderBottom: '1px solid var(--c-panel-border)', paddingBottom: '8px', overflowX: 'auto' }}>
          {mlpState.layers.map((_, i) => (
            <button key={i} onClick={() => { setSelectedLayerIdx(i); setSelectedNeuronIdx(0); }}
               style={{ padding: '6px 12px', background: selectedLayerIdx === i ? 'var(--c-surface-variant)' : 'transparent', color: selectedLayerIdx === i ? 'var(--c-on-surface)' : 'var(--c-on-surface-variant)', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: selectedLayerIdx === i ? 'bold' : 'normal', whiteSpace: 'nowrap' }}>
               {i === mlpState.layers.length - 1 ? 'Output Layer' : `Hidden Layer ${i + 1}`}
            </button>
          ))}
        </div>

        {currentLayer && (
          <>
            <div style={{ marginBottom: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
               {Array.from({ length: currentLayer.nodes }).map((_, i) => (
                 <button key={i} onClick={() => setSelectedNeuronIdx(i)}
                   style={{ padding: '4px 8px', fontSize: '12px', background: selectedNeuronIdx === i ? 'var(--c-primary)' : 'var(--c-surface-container-highest)', color: selectedNeuronIdx === i ? '#fff' : 'var(--c-on-surface)', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                   Neuron {i+1}
                 </button>
               ))}
            </div>
            
            <div className="viz-ctrl__sliders" style={{ padding: '12px 0' }}>
              {currentLayer.weights[selectedNeuronIdx]?.map((w, i) => (
                <div key={`w${i}`} style={{ marginBottom: '12px' }}>
                  <div className="viz-ctrl__slider-row">
                    <label>Weight from Node {i+1}</label>
                    <span className="viz-ctrl__slider-val">{w?.toFixed(3) ?? 'NaN'}</span>
                  </div>
                  <input type="range" className="control__range" min="-3" max="3" step="0.01" value={w ?? 0} onChange={e => handleManualWeight(selectedLayerIdx, selectedNeuronIdx, i, Number(e.target.value))} />
                  <div className="control__range-labels"><span>-3</span><span>3</span></div>
                </div>
              ))}
              <div style={{ marginBottom: '12px' }}>
                <div className="viz-ctrl__slider-row">
                  <label>Bias</label>
                  <span className="viz-ctrl__slider-val">{currentLayer.biases[selectedNeuronIdx]?.toFixed(3) || '0.00'}</span>
                </div>
                <input type="range" className="control__range" min="-3" max="3" step="0.01" value={currentLayer.biases[selectedNeuronIdx] || 0} onChange={e => handleManualBias(selectedLayerIdx, selectedNeuronIdx, Number(e.target.value))} />
                <div className="control__range-labels"><span>-3</span><span>3</span></div>
              </div>
            </div>
          </>
        )}
      </div>

      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">INFERENCE</span>
          <span className="viz-ctrl__subtitle">Forward pass with custom inputs</span>
        </div>
        
        {dataset === 'import' && importStats && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', background: 'rgba(255,255,255,0.03)', padding: '8px 12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)' }}>
            <input
              type="checkbox"
              id="mlp-scale-inputs"
              checked={shouldScaleInputs}
              onChange={e => setShouldScaleInputs(e.target.checked)}
              style={{ cursor: 'pointer', accentColor: 'var(--c-primary)' }}
            />
            <label htmlFor="mlp-scale-inputs" style={{ fontSize: '11px', fontWeight: 500, color: 'var(--c-on-surface-variant)', cursor: 'pointer', userSelect: 'none' }}>
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
            <span className="viz-infer__y" style={{ fontSize: inferResults.length > 0 && inferResults[0].pred.length > 1 ? '14px' : undefined }}>
               {inferResults.length > 0 
                  ? (inferResults[0].pred.length === 1 
                      ? inferResults[0].pred[0].toFixed(3) 
                      : `[${inferResults[0].pred.map(v => v.toFixed(2)).join(', ')}]`)
                  : '—'}
            </span>
          </div>
        </div>
        
        {inferResults.length > 0 && (
          <div className="viz-infer__history" style={{ marginTop: '16px', maxHeight: '180px', overflowY: 'auto' }}>
            <div className="viz-infer__history-header" style={{ display: 'flex', justifyContent: 'space-between', opacity: 0.6, fontSize: '11px', fontWeight: 'bold', borderBottom: '1px solid var(--c-panel-border)', paddingBottom: '6px', marginBottom: '8px' }}>
              <span>Inputs x</span><span>Output ŷ</span>
            </div>
            {inferResults.map((r, i) => {
              const displayFeatures = r.rawFeatures || r.features;
              return (
                <div key={i} className="viz-infer__history-row" style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", color: 'var(--c-on-surface-variant)' }}>[{displayFeatures.map(f => f.toFixed(2)).join(', ')}]</span>
                  <span style={{ 
                    color: r.pred.length === 1 
                      ? (r.pred[0] >= 0.5 ? 'var(--c-primary)' : 'var(--c-tertiary)') 
                      : 'var(--c-primary)', 
                    fontWeight: 'bold',
                    fontFamily: "'JetBrains Mono', monospace"
                  }}>
                    {r.pred.length === 1 
                      ? r.pred[0].toFixed(3) 
                      : `[${r.pred.map(v => v.toFixed(2)).join(', ')}]`}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
      
      {/* 6. BACKPROPAGATION TRACKER */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span className="viz-ctrl__title">BACKPROPAGATION TRACKER</span>
            <span className="viz-ctrl__subtitle">Real-time gradient flow & math equations</span>
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
          {mlpState.layers.map((layer, i) => {
            const isOut = i === mlpState.layers.length - 1;
            const actMath = ACT_FORMULAS[layer.activation] || ACT_FORMULAS.linear;
            const gradAvg = mlpState.layerGradients?.[i] ?? 0;
            const weightAvg = mlpState.layerWeightsMean?.[i] ?? 0;
            
            // Log scale for gradients to make visualization easier (they can be tiny)
            const gradLog = Math.max(0, (Math.log10(gradAvg + 1e-10) + 10) / 10);
            
            return (
              <div key={i} style={{ padding: '12px', background: 'var(--c-surface-variant)', border: '1px solid var(--c-panel-border)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontWeight: 'bold', color: isOut ? 'var(--c-tertiary)' : 'var(--c-primary)' }}>
                    {isOut ? 'Output Layer' : `Hidden Layer ${i + 1}`}
                  </span>
                  <span style={{ fontSize: '10px', background: 'var(--c-surface-container-highest)', padding: '2px 6px' }}>
                    {layer.activation.toUpperCase()}
                  </span>
                </div>
                
                <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', color: 'var(--c-on-surface-variant)', marginBottom: '12px', background: 'rgba(0,0,0,0.2)', padding: '8px', overflowX: 'auto', whiteSpace: 'nowrap' }}>
                  <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Activation Function</div>
                  <div style={{ color: 'var(--c-on-surface)' }}>Forward Pass: <span style={{ color: '#a855f7' }}>{actMath.fn}</span></div>
                  <div style={{ color: 'var(--c-on-surface)' }}>Backward (Deriv): <span style={{ color: '#ec4899' }}>{actMath.deriv}</span></div>
                  
                  <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
                  
                  <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>2. Backpropagation (Chain Rule)</div>
                  {isOut ? (
                    <>
                      <div style={{ color: 'var(--c-on-surface)' }}>Error Term (δ): <span style={{ color: 'var(--c-error)' }}>δₖ = (yₖ - ŷₖ) · f'(sumₖ)</span></div>
                      <div style={{ color: 'var(--c-on-surface)' }}>Weight Update: <span style={{ color: 'var(--c-primary)' }}>Δwⱼₖ = η · δₖ · aⱼ</span></div>
                    </>
                  ) : (
                    <>
                      <div style={{ color: 'var(--c-on-surface)' }}>Error Term (δ): <span style={{ color: 'var(--c-error)' }}>δⱼ = (Σ δₖ·wⱼₖ) · f'(sumⱼ)</span></div>
                      <div style={{ color: 'var(--c-on-surface)' }}>Weight Update: <span style={{ color: 'var(--c-primary)' }}>Δwᵢⱼ = η · δⱼ · aᵢ</span></div>
                    </>
                  )}
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                  <div>
                    <div style={{ fontSize: '10px', opacity: 0.7, marginBottom: '4px' }}>Avg abs(Gradient Δ)</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ flex: 1, height: '24px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                        {mlpState.layerGradientsHistory?.[i] && mlpState.layerGradientsHistory[i].length > 1 ? (
                          <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                            <path 
                              d={mlpState.layerGradientsHistory[i].map((v, idx) => {
                                const x = (idx / Math.max(1, mlpState.layerGradientsHistory![i].length - 1)) * 100;
                                const logV = Math.max(0, (Math.log10(v + 1e-10) + 10) / 10);
                                const y = 24 - Math.min(24, logV * 24);
                                return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                              }).join(' ')} 
                              fill="none" stroke="var(--c-error)" strokeWidth="1.5" strokeLinejoin="round" 
                            />
                          </svg>
                        ) : (
                          <div style={{ width: `${Math.min(100, gradLog * 100)}%`, height: '100%', background: 'var(--c-error)', opacity: 0.5 }} />
                        )}
                      </div>
                      <span style={{ fontSize: '11px', fontFamily: 'monospace', width: '48px', color: 'var(--c-error)' }}>{gradAvg.toExponential(2)}</span>
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '10px', opacity: 0.7, marginBottom: '4px' }}>Avg abs(Weight)</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ flex: 1, height: '24px', background: 'rgba(255,255,255,0.03)', overflow: 'hidden', position: 'relative' }}>
                        {mlpState.layerWeightsMeanHistory?.[i] && mlpState.layerWeightsMeanHistory[i].length > 1 ? (
                          <svg width="100%" height="100%" viewBox="0 0 100 24" preserveAspectRatio="none" style={{ position: 'absolute', inset: 0 }}>
                            <path 
                              d={mlpState.layerWeightsMeanHistory[i].map((v, idx) => {
                                const x = (idx / Math.max(1, mlpState.layerWeightsMeanHistory![i].length - 1)) * 100;
                                const y = 24 - Math.min(24, (v / Math.max(2, weightAvg * 1.5)) * 24); // Dynamically scale Y max
                                return `${idx === 0 ? 'M' : 'L'} ${x} ${y}`;
                              }).join(' ')} 
                              fill="none" stroke="var(--c-primary)" strokeWidth="1.5" strokeLinejoin="round" 
                            />
                          </svg>
                        ) : (
                          <div style={{ width: `${Math.min(100, weightAvg * 20)}%`, height: '100%', background: 'var(--c-primary)', opacity: 0.5 }} />
                        )}
                      </div>
                      <span style={{ fontSize: '11px', fontFamily: 'monospace', width: '40px', color: 'var(--c-primary)' }}>{weightAvg.toFixed(3)}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
      
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-stats__grid">
            <div className="viz-stats__item"><span className="viz-stats__label">N</span><span className="viz-stats__val">{stats.n}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 0</span><span className="viz-stats__val" style={{ color: '#a855f7' }}>{stats.class0}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Class 1</span><span className="viz-stats__val" style={{ color: '#4ade80' }}>{stats.class1}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
