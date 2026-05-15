import { useRef, useEffect, useState, useCallback, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateData, trainStep, predictPerceptron, type Point, type PerceptronState, type ActivationType } from './math';
import { drawDataCanvas, drawNetworkCanvas } from './drawHelpers';

export default function PerceptronVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const netRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const scatterContainerRef = useRef<HTMLDivElement>(null);

  const activation = (params.activation as ActivationType) ?? 'step';
  const lr = (params.learningRate as number) ?? 0.1;
  const maxEp = (params.maxEpochs as number) ?? 100;
  const inputNodes = (params.inputNodes as number) ?? 2;
  const numPerceptrons = (params.numPerceptrons as number) ?? 1;
  
  const numPoints = (datasetParams.points as number) ?? 100;
  const noise = (datasetParams.noise as number) ?? 0.2;

  const [points, setPoints] = useState<Point[]>([]);
  const [percState, setPercState] = useState<PerceptronState>({
    numInputs: inputNodes,
    numPerceptrons: numPerceptrons,
    hiddenWeights: Array.from({ length: numPerceptrons }, () => Array.from({ length: inputNodes }, () => Math.random() - 0.5)),
    hiddenBias: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
    outWeights: Array.from({ length: numPerceptrons }, () => Math.random() - 0.5),
    outBias: Math.random() - 0.5,
    activation,
    learningRate: lr,
    epoch: 0,
    maxEpochs: maxEp,
    lossHistory: [],
    converged: false
  });
  
  const [epochTarget, setEpochTarget] = useState(0);
  const [trained, setTrained] = useState(false);

  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ features: number[]; px: number; py: number } | null>(null);
  
  // Advanced Mode State
  const [advancedMode, setAdvancedMode] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState<'hidden' | 'output'>('hidden');
  const [selectedNeuron, setSelectedNeuron] = useState<number>(0);

  // Inference
  const [inferInputs, setInferInputs] = useState<string[]>(Array(inputNodes).fill('0.50'));
  const [inferResults, setInferResults] = useState<{ features: number[]; pred: number }[]>([]);

  const pushMetrics = useCallback((s: PerceptronState) => {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const p of points) {
      const { pred } = predictPerceptron(p, s);
      const predClass = pred >= 0.5 ? 1 : 0;
      if (p.label === 1 && predClass === 1) tp++;
      else if (p.label === 0 && predClass === 0) tn++;
      else if (p.label === 0 && predClass === 1) fp++;
      else if (p.label === 1 && predClass === 0) fn++;
    }
    
    const total = points.length;
    const acc = total > 0 ? (tp + tn) / total : 0;
    const prec = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const rec = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;

    const loss = s.lossHistory[s.lossHistory.length - 1] ?? 0;
    onMetricsUpdate([
      { label: 'Accuracy', value: (acc * 100).toFixed(1) + '%', isPrimary: true },
      { label: 'Loss (MSE)', value: loss.toFixed(4) },
      { label: 'Epochs', value: String(s.epoch) },
      { label: 'Converged', value: s.converged ? 'Yes' : 'No' },
      { label: 'Precision', value: prec.toFixed(3) },
      { label: 'Recall', value: rec.toFixed(3) },
      { label: 'F1 Score', value: f1.toFixed(3) },
    ]);
  }, [points, onMetricsUpdate]);

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
    if (dataset === 'custom') {
      setPoints([]);
      resetState();
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
    setPercState(s => ({ ...s, activation, learningRate: lr, maxEpochs: maxEp })); 
  }, [activation, lr, maxEp]);

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
    if (dataset === 'custom') return; 
    dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; 
  }, [dataset]);

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
    for (const p of points) {
      const pX = p.features[0]; const pY = p.features[1] ?? 0;
      const d = Math.hypot(pX - px, pY - py);
      if (d < minD) { minD = d; nearest = { features: p.features, px: mx, py: my }; }
    }
    setHoverPt(nearest);

    if (!dragRef.current) return; 
    const dr = dragRef.current; 
    const dx = ((e.clientX - dr.sx) / dw) * (dr.vp.xMax - dr.vp.xMin); 
    const dy = ((e.clientY - dr.sy) / dh) * (dr.vp.yMax - dr.vp.yMin); 
    vpRef.current = { 
      xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, 
      yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy 
    }; 
    setVpVer(v => v + 1); 
  }, [points]);

  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);
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
    if (!isTraining || points.length === 0) return;
    
    let state = percState;
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
      setPercState(state);
      
      if (state.converged || state.epoch >= state.maxEpochs) {
        clearInterval(id);
        setTrained(true);
        onTrainingComplete();
      }
    }, 40);
    
    return () => clearInterval(id);
  }, [isTraining]); // eslint-disable-line
  
  useEffect(() => { pushMetrics(percState); }, [percState, pushMetrics]);

  // Renders
  useEffect(() => { 
    const c = dataRef.current; if (!c) return; 
    const r = () => drawDataCanvas(c, points, percState, vpRef.current, dataset, inferResults); 
    r(); 
    const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); 
    return () => ro.disconnect(); 
  }, [points, percState, vpVer, inferResults, dataset]);
  
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
    const features = inferInputs.map(v => parseFloat(v));
    if (features.some(isNaN)) return;
    const { pred } = predictPerceptron({ features, label: 0 }, percState);
    setInferResults(prev => [{ features, pred }, ...prev].slice(0, 5));
  }, [inferInputs, percState]);

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
    if (points.length < 2) return null;
    let n = points.length, class0 = 0, class1 = 0;
    for (const p of points) {
      if (p.label === 0) class0++; else class1++;
    }
    return { n, class0, class1 };
  })();

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER MAP */}
      <div className="viz-scroll__section viz-scroll__section--canvas" ref={scatterContainerRef}>
        <canvas ref={dataRef} onContextMenu={e => e.preventDefault()} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={() => { handleMouseUp(); setHoverPt(null); }} onClick={dataset === 'custom' ? handleDataClick : undefined} onWheel={handleWheel} style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }} />
        {hoverPt && (
          <div className="viz-tooltip" style={{ left: hoverPt.px + 12, top: hoverPt.py - 8 }}>
            Class: {hoverPt.features[0].toFixed(2)}
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
                    <span className="viz-ctrl__slider-val">{w.toFixed(3)}</span>
                  </div>
                  <input type="range" className="control__range" min="-5" max="5" step="0.01" value={w} onChange={e => handleManualHiddenWeight(selectedNeuron, i, Number(e.target.value))} />
                  <div className="control__range-labels"><span>-5</span><span>5</span></div>
                </div>
              ))}
              <div style={{ marginBottom: '12px' }}>
                <div className="viz-ctrl__slider-row">
                  <label>Bias (b)</label>
                  <span className="viz-ctrl__slider-val">{percState.hiddenBias[selectedNeuron]?.toFixed(3) || '0.00'}</span>
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
                  <span className="viz-ctrl__slider-val">{w.toFixed(3)}</span>
                </div>
                <input type="range" className="control__range" min="-5" max="5" step="0.01" value={w} onChange={e => handleManualOutWeight(i, Number(e.target.value))} />
                <div className="control__range-labels"><span>-5</span><span>5</span></div>
              </div>
            ))}
            <div style={{ marginBottom: '12px' }}>
              <div className="viz-ctrl__slider-row">
                <label>Output Bias (b)</label>
                <span className="viz-ctrl__slider-val">{percState.outBias.toFixed(3)}</span>
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

      {/* 5. INFERENCE */}
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
        
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header">
              <span>Inputs x</span><span>Output ŷ</span>
            </div>
            {inferResults.map((r, i) => (
              <div key={i} className="viz-infer__history-row">
                <span>[{r.features.map(f => f.toFixed(2)).join(', ')}]</span>
                <span style={{ color: r.pred >= 0.5 ? 'var(--c-primary)' : 'var(--c-tertiary)', fontWeight: 'bold' }}>{r.pred.toFixed(3)}</span>
              </div>
            ))}
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
