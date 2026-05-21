import { usePersistentState } from '../../hooks/usePersistentState';
import { useRef, useEffect, useState, useCallback, useMemo, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';
import { generateClassificationData, trainDecisionTree, predictSingle, predictProbability, computeMetrics, computeConfusionMatrix, computeDataStats, getDecisionPath, type Point, type DecisionTreeState } from './math';
import { drawDataCanvas, drawTreeCanvas, drawROCCanvas } from './drawHelpers';
import { usePlayground } from '../../store';

export default function DecisionTreeVisualization({
  params, dataset, datasetParams, isTraining, resetVersion, onTrainingComplete, onMetricsUpdate,
}: VisualizationProps) {
  const dataCanvasRef = useRef<HTMLCanvasElement>(null);
  const treeCanvasRef = useRef<HTMLCanvasElement>(null);
  const rocCanvasRef = useRef<HTMLCanvasElement>(null);

  const [points, setPoints] = usePersistentState<Point[]>('omodels-decision-tree-points', []);
  const [dtState, setDtState] = useState<DecisionTreeState | null>(null);

  const vpRef = useRef({ xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 });
  const [vpVer, setVpVer] = useState(0);
  const dragRef = useRef<{ sx: number; sy: number; vp: typeof vpRef.current } | null>(null);
  const [hoverPt, setHoverPt] = useState<{ x: number; y: number; px: number; py: number } | null>(null);

  const [inferX, setInferX] = useState('0.50');
  const [inferY, setInferY] = useState('0.50');
  const [inferResults, setInferResults] = usePersistentState<{ x: number, y: number, prob: number, cls: number }[]>('omodels-decision-tree-inferResults', []);

  const algorithm = (params.algorithm as 'id3' | 'c45' | 'cart') ?? 'cart';
  const maxDepth = (params.maxDepth as number) ?? 5;
  const minSamplesSplit = (params.minSamplesSplit as number) ?? 2;
  const minSamplesLeaf = (params.minSamplesLeaf as number) ?? 1;
  const numBins = (params.numBins as number) ?? 20;
  const showDecisionPath = (params.showDecisionPath as boolean) ?? true;
  const numPoints = (datasetParams.points as number) ?? 120;
  const noise = (datasetParams.noise as number) ?? 0.2;

  const pushMetrics = useCallback((s: DecisionTreeState | null) => {
    if (s) { onMetricsUpdate(computeMetrics(points, s)); }
    else {
      onMetricsUpdate([
        { label: 'Accuracy', value: '—', isPrimary: true },
        { label: 'Precision', value: '—' }, { label: 'Recall', value: '—' },
        { label: 'F1 Score', value: '—' }, { label: 'Tree Depth', value: '—' },
        { label: 'Nodes', value: '—' },
      ]);
    }
  }, [points, onMetricsUpdate]);

  // Import from store
  const { importedData, importVersion } = usePlayground();
  useEffect(() => {
    if (dataset !== 'import' || !importedData || importedData.length === 0) return;
    // Clamp cls to binary (0 or 1) for binary classifiers
    const pts = (importedData as any[]).map((p: any) => ({
      x: p.x, y: p.y, cls: p.cls >= 1 ? 1 : 0,
    }));
    setPoints(pts);
    // Reset model state for clean start
    setDtState(null); setInferResults([]);
    // Auto-zoom viewport
    const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
    const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
    const xPad = (xMax - xMin) * 0.15 || 0.5, yPad = (yMax - yMin) * 0.15 || 0.5;
    vpRef.current = { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
    setVpVer(v => v + 1);
  }, [importVersion]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (dataset === 'custom' || dataset === 'import') return;
    setPoints(generateClassificationData(dataset, numPoints, noise));
    setDtState(null); setInferResults([]);
  }, [dataset, numPoints, noise]);

  useEffect(() => {
    if (resetVersion === 0) return;
    setDtState(null); setInferResults([]);
    vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 };
    setVpVer(v => v + 1); setHoverPt(null);
  }, [resetVersion]);

  const handleDataClick = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current) return;
    const canvas = dataCanvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect(); const vp = vpRef.current;
    const px = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const py = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    setPoints(prev => [...prev, { x: px, y: py, cls: e.shiftKey ? 1 : 0 }]);
  }, []);

  const handleWheel = useCallback((e: React.WheelEvent<HTMLCanvasElement>) => {
    if (!e.ctrlKey) return; e.preventDefault();
    const f = e.deltaY > 0 ? 1.1 : 0.9;
    const canvas = dataCanvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect(); const vp = vpRef.current;
    const mx = vp.xMin + ((e.clientX - rect.left) / rect.width) * (vp.xMax - vp.xMin);
    const my = vp.yMax - ((e.clientY - rect.top) / rect.height) * (vp.yMax - vp.yMin);
    vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f };
    setVpVer(v => v + 1);
  }, []);

  const handleMouseDown = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    if (dataset === 'custom' || dataset === 'import') return;
    dragRef.current = { sx: e.clientX, sy: e.clientY, vp: { ...vpRef.current } };
  }, [dataset]);

  const handleMouseMove = useCallback((e: RMouseEvent<HTMLCanvasElement>) => {
    const canvas = dataCanvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect(); const vp = vpRef.current;
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
  const resetView = useCallback(() => { vpRef.current = { xMin: -0.1, xMax: 1.1, yMin: -0.1, yMax: 1.1 }; setVpVer(v => v + 1); }, []);
  const zoomBtn = useCallback((f: number) => {
    const vp = vpRef.current; const mx = (vp.xMin + vp.xMax) / 2; const my = (vp.yMin + vp.yMax) / 2;
    vpRef.current = { xMin: mx + (vp.xMin - mx) * f, xMax: mx + (vp.xMax - mx) * f, yMin: my + (vp.yMin - my) * f, yMax: my + (vp.yMax - my) * f };
    setVpVer(v => v + 1);
  }, []);


  // Training
  useEffect(() => {
    if (!isTraining) return;
    if (points.length < 2) { onTrainingComplete(); return; }
    let progress = 0;
    const animId = setInterval(() => {
      progress += 0.15;
      if (progress >= 1.0) {
        clearInterval(animId);
        const newState = trainDecisionTree(points, algorithm, maxDepth, minSamplesSplit, minSamplesLeaf, numBins);
        setDtState(newState);
        pushMetrics(newState);
        onTrainingComplete();
      }
    }, 40);
    return () => clearInterval(animId);
  }, [isTraining]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { pushMetrics(dtState); }, [points]); // eslint-disable-line

  // Draw canvases
  useEffect(() => {
    const canvas = dataCanvasRef.current; if (!canvas) return;
    const render = () => drawDataCanvas(canvas, points, dtState, vpRef.current, inferResults, dataset);
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas); return () => ro.disconnect();
  }, [points, dtState, vpVer, inferResults, dataset]);

  useEffect(() => {
    const canvas = treeCanvasRef.current; if (!canvas) return;
    const render = () => drawTreeCanvas(canvas, dtState);
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas); return () => ro.disconnect();
  }, [dtState]);

  useEffect(() => {
    const canvas = rocCanvasRef.current; if (!canvas) return;
    const render = () => drawROCCanvas(canvas, points, dtState);
    render();
    const ro = new ResizeObserver(() => requestAnimationFrame(render));
    ro.observe(canvas); return () => ro.disconnect();
  }, [points, dtState]);

  const handleInfer = useCallback(() => {
    if (!dtState?.root) return;
    const x = parseFloat(inferX), y = parseFloat(inferY);
    if (isNaN(x) || isNaN(y)) return;
    const cls = predictSingle(x, y, dtState.root);
    const prob = predictProbability(x, y, dtState.root);
    setInferResults(prev => [{ x, y, prob, cls }, ...prev].slice(0, 5));
  }, [inferX, inferY, dtState]);

  const stats = useMemo(() => computeDataStats(points), [points]);
  const cm = useMemo(() => dtState ? computeConfusionMatrix(points, dtState) : { tp: 0, tn: 0, fp: 0, fn: 0 }, [points, dtState]);

  // Decision path for last inference
  const decisionPath = useMemo(() => {
    if (!dtState?.root || inferResults.length === 0 || !showDecisionPath) return null;
    const last = inferResults[0];
    return getDecisionPath(last.x, last.y, dtState.root);
  }, [dtState, inferResults, showDecisionPath]);

  const algoLabel = algorithm === 'id3' ? 'ID3 (Entropy)' : algorithm === 'c45' ? 'C4.5 (Gain Ratio)' : 'CART (Gini)';

  return (
    <div className="viz-scroll">
      {/* 1. SCATTER PLOT */}
      <div className="viz-scroll__section viz-scroll__section--canvas" style={{ position: 'relative' }}>
        <canvas ref={dataCanvasRef}
          className={`viz-canvas ${dataset !== 'custom' ? 'viz-canvas--pan' : 'viz-canvas--draw'}`}
          onContextMenu={e => e.preventDefault()}
          onMouseDown={handleMouseDown} onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
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

      {/* 2. TREE STRUCTURE */}
      <div className="viz-scroll__section viz-scroll__section--loss">
        <canvas ref={treeCanvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* 3. CONFUSION MATRIX & ROC */}
      <div className="viz-scroll__section viz-scroll__section--controls" style={{ display: 'flex', gap: '20px' }}>
        <div style={{ flex: 1 }}>
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">CONFUSION MATRIX</span></div>
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
          <div className="viz-ctrl__header"><span className="viz-ctrl__title">ROC CURVE</span></div>
          <div style={{ height: '120px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)', overflow: 'hidden' }}>
            <canvas ref={rocCanvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
          </div>
        </div>
      </div>

      {/* 4. TREE INFO */}
      {dtState?.root && (
        <div className="viz-scroll__section viz-scroll__section--controls">
          <div className="viz-ctrl__header">
            <span className="viz-ctrl__title">TREE INFO</span>
            <span className="viz-ctrl__subtitle">{algoLabel}</span>
          </div>
          <div className="viz-ctrl__split">
            <div className="viz-ctrl__right" style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-primary)' }}>Structure</h4>
                <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '4px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Total Nodes:</span><span>{dtState.nodeCount}</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Leaf Nodes:</span><span>{dtState.leafCount}</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Tree Depth:</span><span>{dtState.treeDepth}</span></div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Max Depth Limit:</span><span>{dtState.maxDepth}</span></div>
                </div>
              </div>
              <div style={{ background: 'var(--c-surface-container-highest)', padding: '12px', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '12px', color: 'var(--c-tertiary)' }}>Feature Importance</h4>
                <div style={{ fontSize: '11px', color: 'var(--c-on-surface-variant)', display: 'grid', gap: '8px' }}>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span>Feature x₁:</span><span>{(dtState.featureImportance.x * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '4px', background: 'var(--c-surface-variant)', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${dtState.featureImportance.x * 100}%`, background: 'var(--c-primary)', borderRadius: '2px' }} />
                    </div>
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span>Feature x₂:</span><span>{(dtState.featureImportance.y * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: '4px', background: 'var(--c-surface-variant)', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${dtState.featureImportance.y * 100}%`, background: 'var(--c-tertiary)', borderRadius: '2px' }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 5. DECISION TREE ALGORITHM TRACKER */}
      <div className="viz-scroll__section viz-scroll__section--infer">
        <div className="viz-ctrl__header">
          <span className="viz-ctrl__title">DECISION TREE ALGORITHM TRACKER</span>
          <span className="viz-ctrl__subtitle">Mathematical breakdown & Splitting logic</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '12px' }}>
          <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', color: 'var(--c-on-surface-variant)', background: 'rgba(0,0,0,0.2)', padding: '10px', overflowX: 'auto', whiteSpace: 'nowrap' }}>
            {algorithm === 'cart' && (
              <>
                <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Gini Impurity (CART)</div>
                <div style={{ color: 'var(--c-on-surface)' }}>Impurity: <span style={{ color: '#a855f7' }}>G(S) = 1 - Σ (p_i)²</span></div>
                <div style={{ color: 'var(--c-on-surface)' }}>Gini Gain: <span style={{ color: 'var(--c-primary)' }}>ΔG = G(S) - [ (N_L / N)G(S_L) + (N_R / N)G(S_R) ]</span></div>
              </>
            )}
            {algorithm === 'id3' && (
              <>
                <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Entropy & Information Gain (ID3)</div>
                <div style={{ color: 'var(--c-on-surface)' }}>Entropy: <span style={{ color: '#a855f7' }}>H(S) = -Σ p_i log₂(p_i)</span></div>
                <div style={{ color: 'var(--c-on-surface)' }}>Info Gain: <span style={{ color: 'var(--c-primary)' }}>IG = H(S) - [ (N_L / N)H(S_L) + (N_R / N)H(S_R) ]</span></div>
              </>
            )}
            {algorithm === 'c45' && (
              <>
                <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>1. Gain Ratio (C4.5)</div>
                <div style={{ color: 'var(--c-on-surface)' }}>Split Info: <span style={{ color: '#a855f7' }}>SI = - (N_L / N)log₂(N_L / N) - (N_R / N)log₂(N_R / N)</span></div>
                <div style={{ color: 'var(--c-on-surface)' }}>Gain Ratio: <span style={{ color: 'var(--c-primary)' }}>GR = Info Gain / SI</span></div>
              </>
            )}

            <div style={{ height: '1px', background: 'rgba(255,255,255,0.1)', margin: '8px 0' }} />

            <div style={{ color: 'var(--c-on-surface)', marginBottom: '4px', fontWeight: 'bold' }}>2. Splitting Criteria</div>
            <div style={{ color: 'var(--c-on-surface)' }}>Best Split: <span style={{ color: 'var(--c-error)' }}>θ* = argmax (Gain)</span></div>
            <div style={{ color: 'var(--c-on-surface)' }}>Stopping Rules: <span style={{ color: 'var(--c-on-surface-variant)' }}>Depth ≥ {maxDepth}, Samples ≤ {minSamplesSplit}</span></div>
          </div>
        </div>
      </div>

      {/* 6. INFERENCE */}
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
          <button className="viz-infer__btn" onClick={handleInfer} disabled={!dtState}>Predict</button>
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
        {/* Decision Path */}
        {decisionPath && decisionPath.length > 0 && (
          <div style={{ marginTop: '12px', padding: '10px', background: 'var(--c-surface-container-highest)', borderRadius: '4px', border: '1px solid var(--c-panel-border)' }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--c-on-surface-variant)', marginBottom: '6px', letterSpacing: '0.5px' }}>DECISION PATH</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', alignItems: 'center' }}>
              {decisionPath.map((node, i) => (
                <div key={node.id} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <div style={{
                    padding: '3px 8px', borderRadius: node.isLeaf ? '4px' : '12px', fontSize: '10px',
                    background: node.isLeaf ? (node.predictedClass === 1 ? 'var(--c-tertiary)' : 'var(--c-primary)') : 'var(--c-surface-variant)',
                    color: node.isLeaf ? '#000' : 'var(--c-on-surface-variant)',
                    fontWeight: node.isLeaf ? 'bold' : 'normal',
                  }}>
                    {node.isLeaf ? `Class ${node.predictedClass}` : `${node.splitFeature} ≤ ${node.splitValue?.toFixed(2)}`}
                  </div>
                  {i < decisionPath.length - 1 && <span style={{ color: 'var(--c-on-surface-variant)', fontSize: '10px' }}>→</span>}
                </div>
              ))}
            </div>
          </div>
        )}
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
