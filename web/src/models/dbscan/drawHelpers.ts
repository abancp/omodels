import { type DBSCANState, type Point } from './math';

export const CLUSTER_COLORS = [
  '#a855f7', '#e7c365', '#4ade80', '#f87171', '#38bdf8',
  '#fb923c', '#c084fc', '#22d3ee', '#fbbf24', '#a3e635',
  '#ec4899', '#14b8a6', '#6366f1', '#f59e0b', '#8b5cf6'
];

export function drawDataCanvas(
  canvas: HTMLCanvasElement, points: Point[], state: DBSCANState | null,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  hoverPt: { x: number; y: number } | null,
  dataset: string,
  showEpsCircles: boolean,
  showPointTypes: boolean
) {
  const ctx = canvas.getContext('2d'); if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const mapX = (x: number) => ((x - vp.xMin) / (vp.xMax - vp.xMin)) * W;
  const mapY = (y: number) => H - ((y - vp.yMin) / (vp.yMax - vp.yMin)) * H;
  const mapR = (r: number) => (r / (vp.xMax - vp.xMin)) * W; // Assuming aspect ratio ~1
  ctx.clearRect(0, 0, W, H);

  const s = getComputedStyle(document.body);
  const gridColor = s.getPropertyValue('--c-grid').trim() || '#333';
  const textColor = s.getPropertyValue('--c-on-surface-variant').trim() || '#888';

  // Grid
  ctx.strokeStyle = gridColor; ctx.lineWidth = 1; ctx.beginPath();
  const getTicks = (min: number, max: number) => { const step = Math.pow(10, Math.floor(Math.log10((max - min) / 5))); const t = []; for (let v = Math.ceil(min / step) * step; v <= max; v += step) t.push(v); return t; };
  for (const t of getTicks(vp.xMin, vp.xMax)) { const px = mapX(t); ctx.moveTo(px, 0); ctx.lineTo(px, H); }
  for (const t of getTicks(vp.yMin, vp.yMax)) { const py = mapY(t); ctx.moveTo(0, py); ctx.lineTo(W, py); }
  ctx.stroke();
  ctx.fillStyle = textColor; ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = 'top'; for (const t of getTicks(vp.xMin, vp.xMax)) ctx.fillText(t.toFixed(1), mapX(t) + 4, H - 16);
  ctx.textBaseline = 'bottom'; for (const t of getTicks(vp.yMin, vp.yMax)) ctx.fillText(t.toFixed(1), 4, mapY(t) - 4);

  // Draw Eps circles for core points if enabled
  if (state && showEpsCircles) {
    const epsR = mapR(state.eps);
    for (let i = 0; i < points.length; i++) {
      if (state.pointTypes[i] === 'core') {
        const p = points[i];
        const cIdx = state.assignments[i];
        const color = cIdx >= 0 ? CLUSTER_COLORS[cIdx % CLUSTER_COLORS.length] : '#888';
        ctx.beginPath();
        
        if (state.metric === 'manhattan') {
           ctx.moveTo(mapX(p.x), mapY(p.y) - epsR);
           ctx.lineTo(mapX(p.x) + epsR, mapY(p.y));
           ctx.lineTo(mapX(p.x), mapY(p.y) + epsR);
           ctx.lineTo(mapX(p.x) - epsR, mapY(p.y));
           ctx.closePath();
        } else if (state.metric === 'chebyshev') {
           ctx.rect(mapX(p.x) - epsR, mapY(p.y) - epsR, epsR * 2, epsR * 2);
        } else {
           ctx.arc(mapX(p.x), mapY(p.y), epsR, 0, Math.PI * 2);
        }

        ctx.fillStyle = `${color}0C`; ctx.fill();
        ctx.strokeStyle = `${color}1A`; ctx.lineWidth = 1; ctx.stroke();
      }
    }
  }

  // Draw points
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const px = mapX(p.x), py = mapY(p.y);
    
    let color = '#888888';
    let type = 'unassigned';
    if (state) {
      type = state.pointTypes[i];
      const cIdx = state.assignments[i];
      if (type === 'noise') color = '#4b5563'; // gray for noise
      else if (cIdx >= 0) color = CLUSTER_COLORS[cIdx % CLUSTER_COLORS.length];
    }
    
    if (type === 'noise' && showPointTypes) {
      // draw cross for noise
      ctx.beginPath();
      ctx.moveTo(px - 3, py - 3); ctx.lineTo(px + 3, py + 3);
      ctx.moveTo(px + 3, py - 3); ctx.lineTo(px - 3, py + 3);
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
    } else {
      ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
      if (type === 'border' && showPointTypes) {
        ctx.fillStyle = s.getPropertyValue('--c-surface') || '#1e1e1e'; ctx.fill();
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
      } else {
        ctx.fillStyle = color; ctx.fill();
        ctx.strokeStyle = '#00000030'; ctx.lineWidth = 0.5; ctx.stroke();
      }
    }
    
    // Highlight currently expanding point
    if (state && state.phase === 'EXPAND' && i === state.currentPoint) {
       ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2);
       ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
       
       const epsR = mapR(state.eps);
       ctx.beginPath();
       if (state.metric === 'manhattan') {
           ctx.moveTo(px, py - epsR); ctx.lineTo(px + epsR, py); ctx.lineTo(px, py + epsR); ctx.lineTo(px - epsR, py); ctx.closePath();
       } else if (state.metric === 'chebyshev') {
           ctx.rect(px - epsR, py - epsR, epsR * 2, epsR * 2);
       } else {
           ctx.arc(px, py, epsR, 0, Math.PI * 2);
       }
       ctx.fillStyle = `rgba(255,255,255,0.1)`; ctx.fill();
       ctx.strokeStyle = `rgba(255,255,255,0.4)`; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  
  // Hover effect (shows what the eps neighborhood looks like)
  if (hoverPt && (!state || state.phase === 'DONE')) {
    const epsToUse = state ? state.eps : 0.1;
    const metricToUse = state ? state.metric : 'euclidean';
    const epsR = mapR(epsToUse);
    const px = mapX(hoverPt.x), py = mapY(hoverPt.y);
    ctx.beginPath(); 
    if (metricToUse === 'manhattan') {
       ctx.moveTo(px, py - epsR); ctx.lineTo(px + epsR, py); ctx.lineTo(px, py + epsR); ctx.lineTo(px - epsR, py); ctx.closePath();
    } else if (metricToUse === 'chebyshev') {
       ctx.rect(px - epsR, py - epsR, epsR * 2, epsR * 2);
    } else {
       ctx.arc(px, py, epsR, 0, Math.PI * 2);
    }
    ctx.fillStyle = 'rgba(255,255,255,0.05)'; ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1; ctx.stroke();
  }

  if (dataset === 'custom' && points.length < 3) {
    ctx.fillStyle = textColor; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.5; ctx.textAlign = 'center';
    ctx.fillText('Click to add data points', W / 2, H / 2);
    ctx.globalAlpha = 1; ctx.textAlign = 'start';
  }
}
