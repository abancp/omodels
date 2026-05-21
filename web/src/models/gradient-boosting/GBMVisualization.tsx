import { usePersistentState } from '../../hooks/usePersistentState';
import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClassificationData, trainGBM, gbmPredictSingle, gbmPredictProbability, getStageContributions, computeMetrics, computeConfusionMatrix, computeDataStats, type Point, type GBMState } from './math';
import { drawDataCanvas, drawLossCanvas, drawROCCanvas } from './drawHelpers';
import { usePlayground } from '../../store';

export default function GBMVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataRef = useRef<HTMLCanvasElement>(null);
  const lossRef = useRef<HTMLCanvasElement>(null);
  const rocRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = usePersistentState<Point[]>('omodels-gradient-boosting-points', []);
  const [gbmState, setGbmState] = usePersistentState<GBMState | null>('omodels-gradient-boosting-gbmState', null);
  const [activeStages, setActiveStages] = useState<number | undefined>(undefined);
  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);
  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = usePersistentState<{x: number; y: number; prob: number; cls: number; rawScore: number}[]>('omodels-gradient-boosting-inferResults', []);

  const nEstimators = (params.nEstimators as number) ?? 30;
  const learningRate = (params.learningRate as number) ?? 0.1;
  const maxDepth = (params.maxDepth as number) ?? 3;
  const subsample = (params.subsample as number) ?? 0.8;
  const numBins = (params.numBins as number) ?? 15;
  const showStageSlider = (params.showStageSlider as boolean) ?? true;
  const numPoints = (datasetParams.points as number) ?? 150;
  const noise = (datasetParams.noise as number) ?? 0.2;

  const pushMetrics = useCallback((s: GBMState | null) => {
    if (s) onMetricsUpdate(computeMetrics(points, s));
    else onMetricsUpdate([
      { label: 'Accuracy', value: '—', isPrimary: true }, { label: 'Precision', value: '—' },
      { label: 'Recall', value: '—' }, { label: 'F1 Score', value: '—' },
      { label: 'Log Loss', value: '—' }, { label: 'Stages', value: '—' },
    ]);
  }, [points, onMetricsUpdate]);

  // Import from store
  const { importedData, importVersion, testData, testVersion, setTestResults } = usePlayground();

  // Test dataset evaluation
  useEffect(() => {
    if (!testData || testData.length === 0 || !gbmState) return;
    const total = testData.length;
    const results: Record<string, any> = { total, predictions: [] };

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (const p of testData) {
      const x = p.x !== undefined ? p.x : (p.features?.[0] ?? 0);
      const y = p.y !== undefined ? p.y : (p.features?.[1] ?? 0);
      const trueClass = p.cls !== undefined ? p.cls : (p.label ?? 0);
      
      const predClass = gbmPredictSingle(x, y, gbmState);
      const prob = gbmPredictProbability(x, y, gbmState);

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
  }, [testVersion, testData, gbmState]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (dataset !== 'import' || !importedData || importedData.length === 0) return;
    // Clamp cls to binary (0 or 1) for binary classifiers
    const pts = (importedData as any[]).map((p: any) => ({
      x: p.x, y: p.y, cls: p.cls >= 1 ? 1 : 0,
    }));
    setPoints(pts);
    // Reset model state for clean start
    setGbmState(null); setInferResults([]); setActiveStages(undefined);
    // Auto-zoom viewport
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.15 || 0.5, yPad = (yMax - yMin) * 0.15 || 0.5;
    vpRef.current = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
    setVpVer(v => v + 1);
  }, [importVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { if (dataset === 'custom' || dataset === 'import') return; setPoints(generateClassificationData(dataset, numPoints, noise)); setGbmState(null); setInferResults([]); setActiveStages(undefined); }, [dataset, numPoints, noise]);
  useEffect(() => { if (resetVersion === 0) return; setGbmState(null); setInferResults([]); setActiveStages(undefined); vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, [resetVersion]);

  // Mouse handlers (same pattern)
  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dragRef.current) return; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setPoints(prev => [...prev, { x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin), cls: e.shiftKey ? 1 : 0 }]); }, []);
  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => { if (!e.ctrlKey) return; e.preventDefault(); const f = e.deltaY > 0 ? 1.1 : 0.9; const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; const mx = vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin); const my = vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin); vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);
  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { if (dataset === 'custom' || dataset === 'import') return; dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } }; }, [dataset]);
  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => { const c = dataRef.current; if (!c) return; const r = c.getBoundingClientRect(); const vp = vpRef.current; setHoverPt({ x: vp.xMin + ((e.clientX - r.left) / r.width) * (vp.xMax - vp.xMin), y: vp.yMax - ((e.clientY - r.top) / r.height) * (vp.yMax - vp.yMin), px: e.clientX - r.left, py: e.clientY - r.top }); if (!dragRef.current) return; const dr = dragRef.current; const dx = ((e.clientX - dr.sx) / r.width) * (dr.vp.xMax - dr.vp.xMin); const dy = ((e.clientY - dr.sy) / r.height) * (dr.vp.yMax - dr.vp.yMin); vpRef.current = { xMin: dr.vp.xMin - dx, xMax: dr.vp.xMax - dx, yMin: dr.vp.yMin + dy, yMax: dr.vp.yMax + dy }; setVpVer(v => v + 1); }, []);
  const handleMouseUp = useCallback(() => { dragRef.current = null; }, []);
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);
  const zoomBtn = useCallback((f: number) => { const vp = vpRef.current; const mx = (vp.xMin + vp.xMax) / 2; const my = (vp.yMin + vp.yMax) / 2; vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f }; setVpVer(v => v + 1); }, []);

  // Training
  useEffect(() => {
    if (!isTraining) return; if (points.length < 2) { onTrainingComplete(); return; }
    let p = 0;
    const id = setInterval(() => { p += 0.10; if (p >= 1.0) { clearInterval(id); const s = trainGBM(points, nEstimators, learningRate, maxDepth, subsample, numBins); setGbmState(s); setActiveStages(s.stumps.length); pushMetrics(s); onTrainingComplete(); } }, 50);
    return () => clearInterval(id);
  }, [isTraining]); // eslint-disable-line
  useEffect(() => { pushMetrics(gbmState); }, [points]); // eslint-disable-line

  // Canvas renders
  useEffect(() => { const c = dataRef.current; if (!c) return; const r = () => drawDataCanvas(c, points, gbmState, vpRef.current, inferResults, dataset, activeStages); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, gbmState, vpVer, inferResults, dataset, activeStages]);
  useEffect(() => { const c = lossRef.current; if (!c) return; const r = () => drawLossCanvas(c, gbmState); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [gbmState]);
  useEffect(() => { const c = rocRef.current; if (!c) return; const r = () => drawROCCanvas(c, points, gbmState); r(); const ro = new ResizeObserver(() => requestAnimationFrame(r)); ro.observe(c); return () => ro.disconnect(); }, [points, gbmState]);

  const handleInfer = useCallback(() => {
    if (!gbmState) return; const x = parseFloat(inferX), y = parseFloat(inferY); if (isNaN(x) || isNaN(y)) return;
    const cls = gbmPredictSingle(x, y, gbmState); const prob = gbmPredictProbability(x, y, gbmState);
    const rawScore = gbmState.initialPrediction + getStageContributions(x, y, gbmState).reduce((a, b) => a + b, 0);
    setInferResults(prev => [{ x, y, prob, cls, rawScore }, ...prev].slice(0, 5));
  }, [inferX, inferY, gbmState]);

  const stats = useMemo(() => computeDataStats(points), [points]);
  const cm = useMemo(() => gbmState ? computeConfusionMatrix(points, gbmState) : { tp: 0, tn: 0, fp: 0, fn: 0 }, [points, gbmState]);
  const lastContribs = useMemo(() => { if (!gbmState || inferResults.length === 0) return null; return getStageContributions(inferResults[0].x, inferResults[0].y, gbmState); }, [gbmState, inferResults]);

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER */}
      <div className="viz-scroll__section viz-scroll__section--canvas" style={{ position: 'relative' }}>
        <canvas ref={dataRef} onContextMenu={e => e.preventDefault()} onMouseDown={handleMouseDown} onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp} onClick={dataset === 'custom' ? handleDataClick : undefined} onWheel={handleWheel} style={{ width: '100%', height: '100%', display: 'block', cursor: dataset === 'custom' ? 'crosshair' : 'grab' }} />
        {hoverPt && <div className="viz-tooltip" style={{ left: hoverPt.px + 10, top: hoverPt.py - 24 }}>{hoverPt.x.toFixed(2)}, {hoverPt.y.toFixed(2)}</div>}
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

      {/* Stage Slider */}
      {showStageSlider && gbmState && gbmState.stumps.length > 1 && (
        <div className="viz-scroll__section viz-scroll__section--controls" style={{ paddingTop: '8px', paddingBottom: '8px' }}>
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">BOOSTING STAGES</span><span className="viz-ctrl__subtitle">Slide to see boundary evolution</span></div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', minWidth: '20px' }}>1</span>
            <input type="range" min={1} max={gbmState.stumps.length} value={activeStages ?? gbmState.stumps.length} onChange={e => setActiveStages(parseInt(e.target.value))} style={{ flex: 1, accentColor: 'var(--c-primary)' }} />
            <span style={{ fontSize: '11px', color: 'var(--c-primary)', fontWeight: 'bold', minWidth: '40px' }}>{activeStages ?? gbmState.stumps.length} / {gbmState.stumps.length}</span>
          </div>
        </div>
      )}

      {/* 2. LOSS CURVE */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={lossRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. CONFUSION MATRIX & ROC */}
      <div className="viz-scroll__section viz-scroll__section--controls" style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">CONFUSION MATRIX</span></div>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: '4px', textAlign: 'center', fontSize: '11px' }}>
            <div></div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 0</div><div style={{ color: 'var(--c-on-surface-variant)' }}>Pred 1</div>
            <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 0</div>
            <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '14px' }}>{cm.tn}</div><div>TN</div></div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{cm.fp}</div><div>FP</div></div>
            <div style={{ color: 'var(--c-on-surface-variant)', alignSelf: 'center' }}>True 1</div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-error)', fontWeight: 'bold', fontSize: '14px' }}>{cm.fn}</div><div>FN</div></div>
            <div style={{ background: 'var(--c-surface-variant)', padding: '10px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}><div style={{ color: 'var(--c-tertiary)', fontWeight: 'bold', fontSize: '14px' }}>{cm.tp}</div><div>TP</div></div>
          </div>
        </div>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">ROC CURVE</span></div>
          <div style={{ height: '120px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)', overflow: 'hidden' }}>
            <canvas ref={rocRef} style={{ width: '100%', height: '100%', display: 'block' }} />
          </div>
        </div>
      </div>

      {/* 4. GBM INFO */}
      {gbmState && (
        <div className="viz-scroll__section viz-scroll__section--controls">
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">BOOSTING INFO</span><span className="viz-ctrl__subtitle">lr={gbmState.learningRate}, subsample={gbmState.subsample}</span></div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-primary)' }}>Training</h4>
              <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '4px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Stages:</span><span>{gbmState.nEstimators}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Learning Rate:</span><span>{gbmState.learningRate}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Max Depth:</span><span>{gbmState.maxDepth}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Initial Loss:</span><span>{gbmState.lossHistory[0]?.toFixed(4) ?? '—'}</span></div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Final Loss:</span><span style={{ color: 'var(--c-primary)' }}>{gbmState.lossHistory[gbmState.lossHistory.length - 1]?.toFixed(4) ?? '—'}</span></div>
              </div>
            </div>
            <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
              <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-tertiary)' }}>Feature Importance</h4>
              <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '8px' }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}><span>x₁:</span><span>{(gbmState.featureImportance.x * 100).toFixed(1)}%</span></div>
                  <div style={{ height: '4px', background: 'var(--c-surface-variant)', borderRadius: '2px', overflow: 'hidden' }}><div style={{ height: '100%', width: `${gbmState.featureImportance.x * 100}%`, background: 'var(--c-primary)', borderRadius: '2px' }} /></div>
                </div>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}><span>x₂:</span><span>{(gbmState.featureImportance.y * 100).toFixed(1)}%</span></div>
                  <div style={{ height: '4px', background: 'var(--c-surface-variant)', borderRadius: '2px', overflow: 'hidden' }}><div style={{ height: '100%', width: `${gbmState.featureImportance.y * 100}%`, background: 'var(--c-tertiary)', borderRadius: '2px' }} /></div>
                </div>
              </div>
            </div>
          </div>
          {/* Stage contribution mini-bars */}
          <div style={{ marginTop: '12px', padding: '10px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--c-on-surface-variant)', marginBottom: '6px', letterSpacing: '0.5px' }}>STAGE CONTRIBUTIONS (avg magnitude)</div>
            <div style={{ display: 'flex', gap: '1px', height: '24px', alignItems: 'flex-end' }}>
              {gbmState.stageContributions.map((c, i) => {
                const maxC = Math.max(...gbmState.stageContributions);
                return <div key={i} style={{ flex: 1, height: `${maxC > 0 ? (c / maxC) * 100 : 0}%`, minHeight: '1px', background: 'var(--c-primary)', opacity: 0.4 + 0.6 * (c / (maxC || 1)), borderRadius: '1px 1px 0 0' }} />;
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '8px', color: 'var(--c-on-surface-variant)', opacity: 0.5, marginTop: '2px' }}><span>1</span><span>{gbmState.nEstimators}</span></div>
          </div>
        </div>
      )}

      {/* 5. GBM ALGORITHM TRACKER */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">GRADIENT BOOSTING ALGORITHM TRACKER</span>
          <span className="viz-ctrl__subtitle">Mathematical breakdown of stage-wise learning</span>
        </div>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '12px' }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', color: 'var(--c-on-surface-variant)', background: 'rgba(0,0,0,0.2)', padding: '10px', overflowX: 'auto', whiteSpace: 'nowrap' }}>
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Initialization & Objective</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Log-Loss: <span style={{ color: '#a855f7' }}>L(y, F) = -[y·log(p) + (1-y)·log(1-p)]</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Base Score: <span style={{ color: 'var(--c-primary)' }}>F₀(x) = log(P(y=1) / P(y=0))</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>2. Pseudo-Residuals (Gradients)</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Prediction: <span style={{ color: '#a855f7' }}>{"p_m = 1 / (1 + e^{-F_{m-1}(x)})"}</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Residual: <span style={{ color: 'var(--c-error)' }}>r_im = -[∂L / ∂F(x_i)] = y_i - p_m</span></div>
            
            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />
            
            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>3. Model Update (Stage m)</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Fit Weak Learner: <span style={{ color: '#a855f7' }}>h_m(x) ≈ r_im (using Tree Depth {maxDepth})</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Update Score: <span style={{ color: 'var(--c-primary)' }}>{"F_m(x) = F_{m-1}(x) + (η)·h_m(x)"}</span></div>
          </div>
        </div>
      </div>

      {/* 6. INFERENCE */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header"><span className="viz-ctrl__title">INFERENCE</span><span className="viz-ctrl__subtitle">Boosted prediction with stage breakdown</span></div>
        <div className="viz-infer__input-row">
          <div className="viz-infer__field"><label>Input x₁</label><input className="viz-infer__input" type="number" step="0.01" value={inferX} onChange={e => setInferX(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <div className="viz-infer__field"><label>Input x₂</label><input className="viz-infer__input" type="number" step="0.01" value={inferY} onChange={e => setInferY(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleInfer()} style={{ width: '70px' }} /></div>
          <button className="viz-infer__btn" onClick={handleInfer} disabled={!gbmState}>Predict</button>
          <div className="viz-infer__result" style={{ marginLeft: '10px' }}><label>P(y=1)</label><span className="viz-infer__y">{inferResults.length > 0 ? inferResults[0].prob.toFixed(4) : '—'}</span></div>
          <div className="viz-infer__result"><label>Class</label><span className="viz-infer__y" style={{ color: inferResults.length > 0 ? (inferResults[0].cls === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)') : 'inherit' }}>{inferResults.length > 0 ? inferResults[0].cls : '—'}</span></div>
        </div>
        {/* Stage contributions for last inference */}
        {lastContribs && lastContribs.length > 0 && (
          <div style={{ marginTop: '12px', padding: '10px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--c-on-surface-variant)', marginBottom: '6px', letterSpacing: '0.5px' }}>STAGE-WISE CONTRIBUTIONS (raw score: {inferResults[0]?.rawScore.toFixed(3)})</div>
            <div style={{ display: 'flex', gap: '1px', height: '28px', alignItems: 'center' }}>
              {lastContribs.map((c, i) => {
                const maxAbs = Math.max(...lastContribs.map(Math.abs), 0.01);
                const h = Math.abs(c) / maxAbs * 100;
                return <div key={i} style={{ flex: 1, height: `${h}%`, minHeight: '1px', background: c >= 0 ? 'var(--c-tertiary)' : 'var(--c-primary)', opacity: 0.5 + 0.5 * (Math.abs(c) / maxAbs), borderRadius: '1px', alignSelf: c >= 0 ? 'flex-end' : 'flex-start' }} title={`Stage ${i + 1}: ${c.toFixed(4)}`} />;
              })}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '8px', color: 'var(--c-on-surface-variant)', opacity: 0.5, marginTop: '2px' }}><span>Stage 1</span><span>Stage {lastContribs.length}</span></div>
          </div>
        )}
        {inferResults.length > 0 && (
          <div className="viz-infer__history">
            <div className="viz-infer__history-header" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr' }}><span>x₁</span><span>x₂</span><span>P(y=1)</span><span>Class</span><span>Raw</span></div>
            {inferResults.map((r, i) => (
              <div key={i} className="viz-infer__history-row" style={{ gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr' }}>
                <span>{r.x.toFixed(3)}</span><span>{r.y.toFixed(3)}</span><span>{r.prob.toFixed(4)}</span>
                <span style={{ color: r.cls === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)' }}>{r.cls}</span>
                <span>{r.rawScore.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 7. DATA STATISTICS */}
      {stats && (
        <div className="viz-scroll__section viz-scroll__section--stats">
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">DATA STATISTICS</span></div>
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
