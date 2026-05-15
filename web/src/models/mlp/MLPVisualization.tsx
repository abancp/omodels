import { useRef, useEffect, useState, useCallback, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateData, trainStep, predictMLP, type Point, type MLPState, type ActivationType, type MLPLayer } from './math';
import { drawDataCanvas, drawNetworkCanvas } from './drawHelpers';

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
    const outAct = (params.outAct as ActivationType) ?? 'sigmoid';
    layers.push({
      nodes: 1,
      activation: outAct,
      weights: [[...Array(prevNodes)].map(() => (Math.random() - 0.5) * Math.sqrt(2 / prevNodes))],
      biases: [0]
    });
    return layers;
  }, [params, inputNodes, numLayers]);

  const [points, setPoints] = useState<Point[]>([]);
  const [mlpState, setMlpState] = useState<MLPState>(() => ({
    numInputs: inputNodes,
    layers: getLayerConfig(),
    learningRate: lr,
    epoch: 0,
    maxEpochs: maxEp,
    lossHistory: [],
    converged: false
  }));
  
  const [epochTarget, setEpochTarget] = useState(0);
  const [trained, setTrained] = useState(false);

  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  
  // Advanced Mode State
  const [selectedLayerIdx, setSelectedLayerIdx] = useState<number>(0);
  const [selectedNeuronIdx, setSelectedNeuronIdx] = useState<number>(0);

  // Inference
  const [inferInputs, setInferInputs] = useState<string[]>(Array(inputNodes).fill('0.50'));
  const [inferResults, setInferResults] = useState<{ features: number[]; pred: number }[]>([]);

  const pushMetrics = useCallback((s: MLPState) => {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const p of points) {
      const { pred } = predictMLP(p.features, s);
      const predClass = pred >= 0.5 ? 1 : 0;
      if (p.label === 1 && predClass === 1) tp++;
      else if (p.label === 0 && predClass === 0) tn++;
      else if (p.label === 0 && predClass === 1) fp++;
      else if (p.label === 1 && predClass === 0) fn++;
    }
    
    const total = points.length;
    const acc = total > 0 ? (tp + tn) / total : 0;
    
    const loss = s.lossHistory[s.lossHistory.length - 1] ?? 0;
    onMetricsUpdate([
      { label: 'Loss (MSE)', value: loss.toFixed(4), isPrimary: true },
      { label: 'Accuracy', value: (acc * 100).toFixed(1) + '%' },
      { label: 'Epochs', value: String(s.epoch) },
      { label: 'Converged', value: s.converged ? 'Yes' : 'No' },
    ]);
  }, [points, onMetricsUpdate]);

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
    if (dataset === 'custom') {
      setPoints([]);
      resetState();
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
    if (dragRef.current) return; 
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
    if (dataset === 'custom') return; 
    dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; 
  }, [dataset]);

  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { 
    const c = dataRef.current; if (!c) return; 
    const r = c.getBoundingClientRect(); 
    const padL = 20, padT = 20, padR = 20, padB = 20;
    
    if (!dragRef.current) return; 
    const dw = r.width - padL - padR; const dh = r.height - padT - padB;
    const dr = dragRef.current; 
    const dx = ((e.clientX - dr.sx) / dw) * (dr.vp.xMax - dr.vp.xMin); 
    const dy = ((e.clientY - dr.sy) / dh) * (dr.vp.yMax - dr.vp.yMin); 
    vpRef.current = { 
      xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, 
      yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy 
    }; 
    setVpVer(v => v + 1); 
  }, []);

  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);

  // Training Loop
  useEffect(() => {
    if (!isTraining || points.length === 0) return;
    
    let state = mlpState;
    if (trained) {
       state = { ...state, epoch: 0, lossHistory: [], converged: false };
    }
    
    const target = state.lossHistory.length + state.maxEpochs;
    setEpochTarget(target);
    
    const id = setInterval(() => {
      const steps = Math.max(1, Math.floor(state.maxEpochs / 50));
      for (let i = 0; i < steps; i++) {
        state = trainStep(points, state);
        if (state.converged || state.epoch >= state.maxEpochs) break;
      }
      setMlpState(state);
      
      if (state.converged || state.epoch >= state.maxEpochs) {
        clearInterval(id);
        setTrained(true);
        onTrainingComplete();
      }
    }, 40);
    
    return () => clearInterval(id);
  }, [isTraining]); // eslint-disable-line
  
  useEffect(() => { pushMetrics(mlpState); }, [mlpState, pushMetrics]);

  // Renders
  useEffect(() => { 
    const c = dataRef.current; if (!c) return; 
    const r = () => drawDataCanvas(c, points, mlpState, vpRef.current, dataset, inferResults); 
    r(); 
    const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); 
    return () => ro.disconnect(); 
  }, [points, mlpState, vpVer, inferResults, dataset]);
  
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
    const features = inferInputs.map(v => parseFloat(v));
    if (features.some(isNaN)) return;
    const { pred } = predictMLP(features, mlpState);
    setInferResults(prev => [{ features, pred }, ...prev].slice(0, 5));
  }, [inferInputs, mlpState]);

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
    if (points.length < 2) return null;
    let n = points.length, class0 = 0, class1 = 0;
    for (const p of points) {
      if (p.label === 0) class0++; else class1++;
    }
    return { n, class0, class1 };
  })();

  const currentLayer = mlpState.layers[selectedLayerIdx];

  return (
    <div className="viz-scroll">
      <div className="viz-scroll__section viz-scroll__section--canvas" ref={scatterContainerRef}>
        <canvas ref={dataRef} onContextMenu={e => e.preventDefault()} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={() => { handleMouseUp(); }} onClick={dataset === 'custom' ? handleDataClick : undefined} onWheel={handleWheel} style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }} />
        <div className="viz-scatter-ctrls">
          <button className="viz-scatter-btn" onClick={resetView} title="Reset view">⟲</button>
        </div>
      </div>

      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={lossRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      <div className="viz-scroll__section" style={{ minHeight: '300px', position: 'relative' }}>
        <canvas ref={netRef} style={{ width: '100%', height: '100%', display: 'block', borderRadius: '8px', background: 'var(--c-surface-container-low)' }} />
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
                    <span className="viz-ctrl__slider-val">{w.toFixed(3)}</span>
                  </div>
                  <input type="range" className="control__range" min="-3" max="3" step="0.01" value={w} onChange={e => handleManualWeight(selectedLayerIdx, selectedNeuronIdx, i, Number(e.target.value))} />
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
