import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClassificationData, computeMetrics, predictProbabilities, computeConfusionMatrix, computeROCCurve, computeDataStats, trainNaiveBayes, type Point, type NBState } from './math';

export default function NaiveBayesVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataCanvasRef = useRef<HTMLCanvasElement>(null);
  const rocCanvasRef = useRef<HTMLCanvasElement>(null);

  // State
  const [points, setPoints] = useState<Point[]>([]);
  const [nbState, setNbState] = useState<NBState | null>(null);

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
  const nbType = (params.nbType as 'gaussian' | 'multinomial' | 'bernoulli') ?? 'gaussian';
  const varSmoothing = (params.varSmoothing as number) ?? 1e-9;
  const alpha = (params.alpha as number) ?? 1.0;
  const binarizeThreshold = (params.binarizeThreshold as number) ?? 0.5;
  const fitPrior = (params.fitPrior as boolean) ?? true;
  
  const numPoints = (datasetParams.points as number) ?? 100;
  const noise = (datasetParams.noise as number) ?? 0.15;

  const pushMetrics = useCallback((state: NBState | null) => {
    if (state) {
      onMetricsUpdate(computeMetrics(points, state));
    } else {
      onMetricsUpdate([
        { label: 'Accuracy', value: '—', isPrimary: true },
        { label: 'Precision', value: '—' },
        { label: 'Recall', value: '—' },
        { label: 'F1 Score', value: '—' },
        { label: 'AUC', value: '—' },
      ]);
    }
  }, [points, onMetricsUpdate]);

  /* Generate dataset */
  useEffect(() => {
    if (dataset === 'custom') return;
    const pts = generateClassificationData(dataset, numPoints, noise);
    setPoints(pts);
    setNbState(null);
  }, [dataset, numPoints, noise]);

  /* Full reset */
  useEffect(() => {
    if (resetVersion === 0) return;
    setNbState(null);
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1);
    setHoverPt(null);
    setInferResults([]);
  }, [resetVersion]);

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
    if (!e.ctrlKey || dataset === 'custom') return;
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

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.getElementById('canvas-area')?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }, []);

  /* Training (Instant for NB) */
  useEffect(() => {
    if (!isTraining) return;
    if (points.length < 2) {
      onTrainingComplete();
      return;
    }

    // Naive Bayes is an exact method, no iterations needed
    // Simulate a brief animation to fit the UI expectation
    let progress = 0;
    const animId = setInterval(() => {
      progress += 0.2;
      if (progress >= 1.0) {
        clearInterval(animId);
        const newState = trainNaiveBayes(points, nbType, fitPrior, varSmoothing, alpha, binarizeThreshold);
        setNbState(newState);
        pushMetrics(newState);
        onTrainingComplete();
      }
    }, 50);

    return () => clearInterval(animId);
  }, [isTraining]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { pushMetrics(nbState); }, [points]); // eslint-disable-line

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
    if (nbState) {
      const res = 6; 
      for (let px = 0; px < W; px += res) {
        for (let py = 0; py < H; py += res) {
          const nx = vp.xMin + ((px + res/2) / W) * (vp.xMax - vp.xMin);
          const ny = vp.yMax - ((py + res/2) / H) * (vp.yMax - vp.yMin);
          
          const probs = predictProbabilities(nx, ny, nbState);
          const prob1 = probs[1] || 0;
          
          if (prob1 >= 0.5) {
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
          const probs = predictProbabilities(nx, ny, nbState);
          const prob1 = probs[1] || 0;
          
          if (Math.abs(prob1 - 0.5) < 0.015) {
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
  }, [points, nbState, vpVer, inferResults]);

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

    if (nbState) {
      const { curve, auc } = computeROCCurve(points, nbState);
      if (curve.length > 0) {
        ctx.beginPath();
        for (let i = 0; i < curve.length; i++) {
          const x = padL + curve[i].fpr * cw;
          const y = h - padB - curve[i].tpr * ch;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = primary; ctx.lineWidth = 2; ctx.stroke();
      }
      ctx.fillStyle = primary; ctx.font = "600 10px 'Inter', sans-serif";
      ctx.fillText(`AUC: ${auc.toFixed(3)}`, w - 50, padT + 10);
    } else {
      ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
      ctx.fillText('Train model to see ROC', w / 2, h / 2 + 10);
      ctx.globalAlpha = 1; ctx.textAlign = 'start';
    }

    ctx.fillStyle = muted; ctx.font = "9px 'Inter', sans-serif";
    ctx.fillText('FPR', w / 2, h - 5);
    ctx.save(); ctx.translate(10, h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('TPR', -10, 0); ctx.restore();
  }, [points, nbState]);

  const handleInfer = useCallback(() => {
    if (!nbState) return;
    const x = parseFloat(inferX);
    const y = parseFloat(inferY);
    if (isNaN(x) || isNaN(y)) return;
    const probs = predictProbabilities(x, y, nbState);
    const prob = probs[1] || 0;
    const cls = prob >= 0.5 ? 1 : 0;
    setInferResults(prev => [{x, y, prob, cls}, ...prev].slice(0, 5));
  }, [inferX, inferY, nbState]);

  const stats = useMemo(() => computeDataStats(points), [points]);
  const cm = useMemo(() => nbState ? computeConfusionMatrix(points, nbState) : {tp:0, tn:0, fp:0, fn:0}, [points, nbState]);

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
          <button className="viz-scatter-btn" onClick={toggleFullscreen} title="Fullscreen">⛶</button>
        </div>
      </div>

      {/* 2. NO LOSS CURVE */}
      <div className="viz-scroll__section viz-scroll__section--loss" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ color: 'var(--c-on-surface-variant)', fontSize: '13px', textAlign: 'center', opacity: 0.6 }}>
          Naive Bayes models are trained analytically using exact probability distributions, so there is no iterative loss curve.
        </div>
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

      {/* 4. NB PARAMETERS */}
      {nbState && (
        <div className="viz-scroll__section viz-scroll__section--controls">
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">MODEL PARAMETERS</span>
            <span className="viz-ctrl__subtitle">Calculated Priors and Likelihoods</span>
          </div>
          <div className="viz-ctrl__split">
            <div className="viz-ctrl__right" style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-primary)' }}>Class 0</h4>
                <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Prior Probability:</span>
                    <span>{Math.exp(nbState.classPrior[0] || 0).toFixed(3)}</span>
                  </div>
                  {nbType === 'gaussian' && nbState.theta && nbState.var && (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Mean x₁:</span>
                        <span>{nbState.theta[0][0].toFixed(3)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Variance x₁:</span>
                        <span>{nbState.var[0][0].toFixed(4)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Mean x₂:</span>
                        <span>{nbState.theta[0][1].toFixed(3)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Variance x₂:</span>
                        <span>{nbState.var[0][1].toFixed(4)}</span>
                      </div>
                    </>
                  )}
                  {nbType !== 'gaussian' && nbState.featureLogProb && (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>LogProb x₁:</span>
                        <span>{nbState.featureLogProb[0][0].toFixed(3)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>LogProb x₂:</span>
                        <span>{nbState.featureLogProb[0][1].toFixed(3)}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-tertiary)' }}>Class 1</h4>
                <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <span>Prior Probability:</span>
                    <span>{Math.exp(nbState.classPrior[1] || 0).toFixed(3)}</span>
                  </div>
                  {nbType === 'gaussian' && nbState.theta && nbState.var && (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Mean x₁:</span>
                        <span>{nbState.theta[1][0].toFixed(3)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Variance x₁:</span>
                        <span>{nbState.var[1][0].toFixed(4)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Mean x₂:</span>
                        <span>{nbState.theta[1][1].toFixed(3)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Variance x₂:</span>
                        <span>{nbState.var[1][1].toFixed(4)}</span>
                      </div>
                    </>
                  )}
                  {nbType !== 'gaussian' && nbState.featureLogProb && (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>LogProb x₁:</span>
                        <span>{nbState.featureLogProb[1][0].toFixed(3)}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>LogProb x₂:</span>
                        <span>{nbState.featureLogProb[1][1].toFixed(3)}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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
          <button className="viz-infer__btn" onClick={handleInfer} disabled={!nbState}>Predict</button>
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
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₁</span><span className="viz-stats__val">{stats.meanX.toFixed(3)}</span></div>
            <div className="viz-stats__item"><span className="viz-stats__label">Mean x₂</span><span className="viz-stats__val">{stats.meanY.toFixed(3)}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
