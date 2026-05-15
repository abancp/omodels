import { useRef, useEffect, useState, useCallback, type MouseEvent as RMouseEvent } from 'react';
import type { VisualizationProps } from '../registry';

type ActivationFn = {
  name: string;
  func: (x: number) => number;
  deriv: (x: number) => number;
  formula: string;
  yDomain: [number, number];
};

const ACTIVATIONS: Record<string, ActivationFn> = {
  step: { name: 'Step (Binary)', func: x => x >= 0 ? 1 : 0, deriv: () => 0, formula: 'y = 1 if x ≥ 0 else 0', yDomain: [-0.5, 1.5] },
  linear: { name: 'Linear', func: x => x, deriv: () => 1, formula: 'y = x', yDomain: [-5, 5] },
  sigmoid: { name: 'Sigmoid', func: x => 1 / (1 + Math.exp(-x)), deriv: x => { const s = 1/(1+Math.exp(-x)); return s*(1-s); }, formula: 'y = 1 / (1 + e^-x)', yDomain: [-0.2, 1.2] },
  tanh: { name: 'Tanh', func: x => Math.tanh(x), deriv: x => 1 - Math.tanh(x)**2, formula: 'y = (e^x - e^-x) / (e^x + e^-x)', yDomain: [-1.5, 1.5] },
  relu: { name: 'ReLU', func: x => Math.max(0, x), deriv: x => x > 0 ? 1 : 0, formula: 'y = max(0, x)', yDomain: [-1, 5] },
  leaky_relu: { name: 'Leaky ReLU', func: x => x > 0 ? x : 0.1 * x, deriv: x => x > 0 ? 1 : 0.1, formula: 'y = x if x > 0 else 0.1x', yDomain: [-1, 5] },
  elu: { name: 'ELU', func: x => x >= 0 ? x : Math.exp(x) - 1, deriv: x => x >= 0 ? 1 : Math.exp(x), formula: 'y = x if x ≥ 0 else α(e^x - 1)', yDomain: [-1.5, 5] },
  swish: { name: 'Swish', func: x => x / (1 + Math.exp(-x)), deriv: x => { const s = 1/(1+Math.exp(-x)); return s + x*s*(1-s); }, formula: 'y = x * sigmoid(x)', yDomain: [-1, 5] },
};

export default function ActivationsVisualization({
  params, onMetricsUpdate,
}: VisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const derivCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const [hoverX, setHoverX] = useState<number | null>(null);
  const activeKey = (params.function as string) ?? 'sigmoid';
  const showDerivative = (params.showDerivative as boolean) ?? true;
  
  const act = ACTIVATIONS[activeKey] || ACTIVATIONS['sigmoid'];

  useEffect(() => {
    onMetricsUpdate([
      { label: 'Equation', value: act.formula, isPrimary: true },
    ]);
  }, [act, onMetricsUpdate]);

  const drawPlot = useCallback((canvas: HTMLCanvasElement, isDeriv: boolean) => {
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;
    
    ctx.clearRect(0, 0, W, H);
    
    const root = getComputedStyle(document.documentElement);
    const primary = root.getPropertyValue(isDeriv ? '--c-tertiary' : '--c-primary').trim() || (isDeriv ? '#e7c365' : '#cfbcff');
    const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
    const border = root.getPropertyValue('--c-panel-border').trim() || '#ffffff20';
    
    // Domain
    const xMin = -5, xMax = 5;
    const yDomain = act.yDomain;
    let yMin = yDomain[0], yMax = yDomain[1];
    
    if (isDeriv) {
      yMin = -0.5; yMax = 1.5;
      if (activeKey === 'relu' || activeKey === 'linear') { yMin = -0.2; yMax = 1.2; }
    }

    const pad = 30;
    const mapX = (x: number) => pad + ((x - xMin) / (xMax - xMin)) * (W - pad * 2);
    const mapY = (y: number) => H - pad - ((y - yMin) / (yMax - yMin)) * (H - pad * 2);

    // Axes
    ctx.strokeStyle = border; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, mapY(0)); ctx.lineTo(W - pad, mapY(0)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(mapX(0), pad); ctx.lineTo(mapX(0), H - pad); ctx.stroke();

    // Labels
    ctx.fillStyle = muted; ctx.font = "10px monospace"; ctx.textAlign = "center";
    ctx.fillText("x=0", mapX(0), H - pad + 15);
    ctx.fillText("-5", mapX(-5), H - pad + 15);
    ctx.fillText("5", mapX(5), H - pad + 15);
    
    ctx.textAlign = "right";
    ctx.fillText("0", pad - 5, mapY(0) + 3);
    ctx.fillText(yMax.toFixed(1), pad - 5, mapY(yMax) + 3);
    ctx.fillText(yMin.toFixed(1), pad - 5, mapY(yMin) + 3);

    // Title
    ctx.fillStyle = muted; ctx.font = "12px Inter"; ctx.textAlign = "left";
    ctx.fillText(isDeriv ? 'Derivative f\'(x)' : 'Function f(x)', pad, pad - 10);

    // Plot Line
    ctx.beginPath();
    ctx.strokeStyle = primary; ctx.lineWidth = 3;
    const steps = 300;
    for (let i = 0; i <= steps; i++) {
      const x = xMin + (i / steps) * (xMax - xMin);
      const y = isDeriv ? act.deriv(x) : act.func(x);
      const px = mapX(x), py = mapY(y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Hover Dot
    if (hoverX !== null) {
      const y = isDeriv ? act.deriv(hoverX) : act.func(hoverX);
      const px = mapX(hoverX), py = mapY(y);
      
      ctx.setLineDash([4, 4]); ctx.strokeStyle = '#ffffff60'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, pad); ctx.lineTo(px, H - pad); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pad, py); ctx.lineTo(W - pad, py); ctx.stroke();
      ctx.setLineDash([]);
      
      ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = primary; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
      
      ctx.fillStyle = '#fff'; ctx.textAlign = hoverX > 0 ? 'right' : 'left';
      ctx.fillText(`${isDeriv ? "f'" : "f"}(${hoverX.toFixed(2)}) = ${y.toFixed(3)}`, px + (hoverX > 0 ? -10 : 10), py - 10);
    }
  }, [act, activeKey, hoverX]);

  useEffect(() => { const c = canvasRef.current; if (c) drawPlot(c, false); }, [drawPlot]);
  useEffect(() => { const c = derivCanvasRef.current; if (c && showDerivative) drawPlot(c, true); }, [drawPlot, showDerivative]);

  const handleMouseMove = (e: RMouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const pad = 30;
    const xMin = -5, xMax = 5;
    let x = xMin + ((e.clientX - rect.left - pad) / (rect.width - pad * 2)) * (xMax - xMin);
    x = Math.max(xMin, Math.min(xMax, x));
    setHoverX(x);
  };

  return (
    <div className="viz-scroll">
      <div className="viz-scroll__section viz-scroll__section--canvas" style={{ position: 'relative' }}>
        <canvas ref={canvasRef} 
          onMouseMove={handleMouseMove} 
          onMouseLeave={() => setHoverX(null)} 
          style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair' }} 
        />
        <div className="viz-scatter-ctrls">
          <button className="viz-scatter-btn" onClick={() => { const c = canvasRef.current?.parentElement; if(c){ if(document.fullscreenElement) document.exitFullscreen(); else c.requestFullscreen(); } }} title="Full Screen">⛶</button>
        </div>
      </div>
      
      {showDerivative && (
        <div className="viz-scroll__section viz-scroll__section--loss" style={{ position: 'relative' }}>
          <canvas ref={derivCanvasRef} 
            onMouseMove={handleMouseMove} 
            onMouseLeave={() => setHoverX(null)} 
            style={{ width: '100%', height: '100%', display: 'block', cursor: 'crosshair' }} 
          />
        </div>
      )}
    </div>
  );
}
