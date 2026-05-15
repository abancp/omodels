import { type GMMState, type Point, getCovarianceEllipse } from './math';

export const CLUSTER_COLORS = [
  '#a855f7', '#e7c365', '#4ade80', '#f87171', '#38bdf8',
  '#fb923c', '#c084fc', '#22d3ee', '#fbbf24', '#a3e635'
];

function hexToRgb(hex: string) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 0, g: 0, b: 0 };
}

export function drawDataCanvas(
  canvas: HTMLCanvasElement, points: Point[], state: GMMState | null,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  inferResults: { x: number; y: number; probs: number[] }[],
  dataset: string,
  showCovariance: boolean,
  colorMixing: boolean
) {
  const ctx = canvas.getContext('2d'); if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const mapX = (x: number) => ((x - vp.xMin) / (vp.xMax - vp.xMin)) * W;
  const mapY = (y: number) => H - ((y - vp.yMin) / (vp.yMax - vp.yMin)) * H;
  const scale = W / (vp.xMax - vp.xMin); // pixels per coordinate unit
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

  // Draw Covariance Ellipses
  if (state && showCovariance) {
    for (let j = 0; j < state.k; j++) {
      const cov = state.covariances[j];
      const mean = state.means[j];
      const color = CLUSTER_COLORS[j % CLUSTER_COLORS.length];
      const { angle, r1, r2 } = getCovarianceEllipse(cov);
      
      const cx = mapX(mean.x);
      const cy = mapY(mean.y);
      
      // Draw 1, 2, and 3 sigma contours
      for (let sigma = 1; sigma <= 3; sigma++) {
        ctx.beginPath();
        ctx.ellipse(cx, cy, r1 * scale * sigma, r2 * scale * sigma, -angle, 0, 2 * Math.PI);
        ctx.strokeStyle = `${color}${sigma === 1 ? '80' : sigma === 2 ? '40' : '15'}`;
        ctx.lineWidth = sigma === 1 ? 2 : 1;
        ctx.stroke();
        
        if (sigma === 1) {
          ctx.fillStyle = `${color}0A`;
          ctx.fill();
        }
      }
      
      // Center crosshair
      ctx.beginPath();
      ctx.moveTo(cx - 4, cy); ctx.lineTo(cx + 4, cy);
      ctx.moveTo(cx, cy - 4); ctx.lineTo(cx, cy + 4);
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
    }
  }

  // Draw Points
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const px = mapX(p.x), py = mapY(p.y);
    
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2);
    
    if (state && state.iteration > 0) {
      if (colorMixing) {
        // Blend colors based on responsibilities
        let r = 0, g = 0, b = 0;
        for (let j = 0; j < state.k; j++) {
          const w = state.responsibilities[i][j];
          const rgb = hexToRgb(CLUSTER_COLORS[j % CLUSTER_COLORS.length]);
          r += rgb.r * w; g += rgb.g * w; b += rgb.b * w;
        }
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      } else {
        // Hard assignment color
        const maxIdx = state.assignments[i];
        ctx.fillStyle = CLUSTER_COLORS[maxIdx % CLUSTER_COLORS.length];
      }
    } else {
      ctx.fillStyle = '#888888';
    }
    
    ctx.fill();
    ctx.strokeStyle = '#00000030'; ctx.lineWidth = 0.5; ctx.stroke();
  }

  // Draw Inference Points
  for (const ir of inferResults) {
    const sx = mapX(ir.x), sy = mapY(ir.y);
    
    // Blend colors
    let r = 0, g = 0, b = 0;
    for (let j = 0; j < ir.probs.length; j++) {
      const w = ir.probs[j];
      const rgb = hexToRgb(CLUSTER_COLORS[j % CLUSTER_COLORS.length]);
      r += rgb.r * w; g += rgb.g * w; b += rgb.b * w;
    }
    const color = `rgb(${r}, ${g}, ${b})`;
    
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2);
    ctx.strokeStyle = '#ffffff80'; ctx.lineWidth = 1.5; ctx.stroke();
  }

  if (dataset === 'custom' && points.length < 3) {
    ctx.fillStyle = textColor; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.5; ctx.textAlign = 'center';
    ctx.fillText('Click to add data points', W / 2, H / 2);
    ctx.globalAlpha = 1; ctx.textAlign = 'start';
  }
}

export function drawLikelihoodCanvas(canvas: HTMLCanvasElement, state: GMMState | null) {
  const ctx = canvas.getContext('2d'); if (!ctx) return;
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

  ctx.fillStyle = muted; ctx.font = "600 10px 'Inter', sans-serif"; ctx.globalAlpha = 0.6; ctx.fillText('LOG-LIKELIHOOD CONVERGENCE', 12, 18); ctx.globalAlpha = 1;

  if (!state || state.logLikelihoodHistory.length < 2) {
    ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
    ctx.fillText('Train to see likelihood curve', w / 2, h / 2 + 10); ctx.globalAlpha = 1; ctx.textAlign = 'start'; return;
  }

  const padL = 45, padR = 12, padT = 30, padB = 24;
  const cw = w - padL - padR, ch = h - padT - padB;
  ctx.strokeStyle = border; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

  const hist = state.logLikelihoodHistory.slice(1); // Skip initial -Infinity if present
  if (hist.length < 2) return;
  
  const minV = Math.min(...hist);
  const maxV = Math.max(...hist);
  const range = Math.max(maxV - minV, 1);

  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const x = padL + (i / (hist.length - 1)) * cw;
    const y = padT + ((maxV - hist[i]) / range) * ch;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = primary; ctx.lineWidth = 1.5; ctx.stroke();
  
  // Fill area
  const lastX = padL + ((hist.length - 1) / (hist.length - 1)) * cw;
  const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
  gradient.addColorStop(0, `${primary}20`); gradient.addColorStop(1, `${primary}02`);
  ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();

  ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
  ctx.fillText('0', padL - 6, h - padB + 12); ctx.fillText(String(hist.length), w - padR - 10, h - padB + 12);
  ctx.fillText(maxV.toFixed(1), 2, padT + 4); 
  ctx.fillText(minV.toFixed(1), 2, h - padB);
  ctx.globalAlpha = 1;
}
