/**
 * Canvas drawing helpers for Gradient Boosting visualization.
 */
import { type GBMState, type Point, gbmPredictSingle, gbmPredictProbability, computeROCCurve } from './math';

/* ─── Draw Decision Boundary ─── */
export function drawDataCanvas(
  canvas: HTMLCanvasElement, points: Point[], state: GBMState | null,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  inferResults: { x: number; y: number; cls: number }[], dataset: string,
  activeStages?: number
) {
  const ctx = canvas.getContext('2d'); if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const mapX = (x: number) => ((x - vp.xMin) / (vp.xMax - vp.xMin)) * W;
  const mapY = (y: number) => H - ((y - vp.yMin) / (vp.yMax - vp.yMin)) * H;
  ctx.clearRect(0, 0, W, H);

  const s = getComputedStyle(document.body);
  const gridColor = s.getPropertyValue('--c-grid').trim() || '#333';
  const textColor = s.getPropertyValue('--c-on-surface-variant').trim() || '#888';
  const primary = s.getPropertyValue('--c-primary').trim() || '#a855f7';
  const tertiary = s.getPropertyValue('--c-tertiary').trim() || '#e7c365';

  if (state && state.stumps.length > 0) {
    const res = 6; const stages = activeStages ?? state.stumps.length;
    for (let px = 0; px < W; px += res) {
      for (let py = 0; py < H; py += res) {
        const nx = vp.xMin + ((px + res / 2) / W) * (vp.xMax - vp.xMin);
        const ny = vp.yMax - ((py + res / 2) / H) * (vp.yMax - vp.yMin);
        const pred = gbmPredictSingle(nx, ny, state, stages);
        ctx.fillStyle = pred === 1 ? `${tertiary}15` : `${primary}15`;
        ctx.fillRect(px, py, res, res);
      }
    }
    for (let px = 0; px < W; px += 3) {
      for (let py = 0; py < H; py += 3) {
        const nx = vp.xMin + (px / W) * (vp.xMax - vp.xMin);
        const ny = vp.yMax - (py / H) * (vp.yMax - vp.yMin);
        const prob = gbmPredictProbability(nx, ny, state, stages);
        if (Math.abs(prob - 0.5) < 0.02) { ctx.fillStyle = '#ffffff50'; ctx.fillRect(px, py, 3, 3); }
      }
    }
  }

  ctx.strokeStyle = gridColor; ctx.lineWidth = 1; ctx.beginPath();
  const getTicks = (min: number, max: number) => { const step = Math.pow(10, Math.floor(Math.log10((max - min) / 5))); const t = []; for (let v = Math.ceil(min / step) * step; v <= max; v += step) t.push(v); return t; };
  for (const t of getTicks(vp.xMin, vp.xMax)) { const px = mapX(t); ctx.moveTo(px, 0); ctx.lineTo(px, H); }
  for (const t of getTicks(vp.yMin, vp.yMax)) { const py = mapY(t); ctx.moveTo(0, py); ctx.lineTo(W, py); }
  ctx.stroke();
  ctx.fillStyle = textColor; ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = 'top'; for (const t of getTicks(vp.xMin, vp.xMax)) ctx.fillText(t.toFixed(1), mapX(t) + 4, H - 16);
  ctx.textBaseline = 'bottom'; for (const t of getTicks(vp.yMin, vp.yMax)) ctx.fillText(t.toFixed(1), 4, mapY(t) - 4);

  for (const p of points) { ctx.beginPath(); ctx.arc(mapX(p.x), mapY(p.y), 5, 0, Math.PI * 2); ctx.fillStyle = p.cls === 0 ? primary : tertiary; ctx.fill(); ctx.strokeStyle = '#00000040'; ctx.lineWidth = 1; ctx.stroke(); }
  for (const ir of inferResults) { const sx = mapX(ir.x), sy = mapY(ir.y); ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2); ctx.fillStyle = ir.cls === 0 ? primary : tertiary; ctx.globalAlpha = 0.9; ctx.fill(); ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(sx, sy, 9, 0, Math.PI * 2); ctx.strokeStyle = ir.cls === 0 ? `${primary}60` : `${tertiary}60`; ctx.lineWidth = 1; ctx.stroke(); }
  if (dataset === 'custom' && points.length < 3) { ctx.fillStyle = textColor; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.5; ctx.textAlign = 'center'; ctx.fillText('Click to add points (Shift+Click for class 1)', W / 2, H / 2); ctx.globalAlpha = 1; ctx.textAlign = 'start'; }
}

/* ─── Draw Loss Curve ─── */
export function drawLossCanvas(canvas: HTMLCanvasElement, state: GBMState | null) {
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

  ctx.fillStyle = muted; ctx.font = "600 10px 'Inter', sans-serif"; ctx.globalAlpha = 0.6; ctx.fillText('LOG LOSS CURVE', 12, 18); ctx.globalAlpha = 1;

  if (!state || state.lossHistory.length < 1) {
    ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
    ctx.fillText('Train to see loss curve', w / 2, h / 2 + 10); ctx.globalAlpha = 1; ctx.textAlign = 'start'; return;
  }

  const padL = 40, padR = 16, padT = 30, padB = 24;
  const cw = w - padL - padR, ch = h - padT - padB;
  ctx.strokeStyle = border; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

  const loss = state.lossHistory;
  const maxL = Math.max(...loss) * 1.1 || 1;
  const totalEp = loss.length;

  ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
  for (let i = 0; i <= 4; i++) { const val = maxL * (1 - i / 4); const y = padT + (i / 4) * ch; ctx.fillText(val.toFixed(3), 2, y + 3); ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.strokeStyle = border; ctx.lineWidth = 0.3; ctx.stroke(); }
  for (let i = 0; i <= 4; i++) { const ep = Math.round((totalEp / 4) * i); const x = padL + (i / 4) * cw; ctx.fillText(String(ep), x - 6, h - 8); }
  ctx.globalAlpha = 1;

  ctx.beginPath();
  for (let i = 0; i < loss.length; i++) { const x = padL + (i / (totalEp - 1)) * cw; const y = padT + ((maxL - loss[i]) / maxL) * ch; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
  ctx.strokeStyle = primary; ctx.lineWidth = 1.5; ctx.stroke();

  const lastX = padL + ((loss.length - 1) / (totalEp - 1)) * cw;
  const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
  gradient.addColorStop(0, `${primary}20`); gradient.addColorStop(1, `${primary}02`);
  ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
}

/* ─── Draw ROC ─── */
export function drawROCCanvas(canvas: HTMLCanvasElement, points: Point[], state: GBMState | null) {
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
  const bdr = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';
  const padL = 30, padB = 20, padT = 10, padR = 10, cw = w - padL - padR, ch = h - padT - padB;
  ctx.strokeStyle = bdr; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, padT); ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);
  if (state && state.stumps.length > 0) {
    const { curve, auc } = computeROCCurve(points, state);
    if (curve.length > 0) { ctx.beginPath(); for (let i = 0; i < curve.length; i++) { const x = padL + curve[i].fpr * cw, y = h - padB - curve[i].tpr * ch; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); } ctx.strokeStyle = primary; ctx.lineWidth = 2; ctx.stroke(); }
    ctx.fillStyle = primary; ctx.font = "600 10px 'Inter', sans-serif"; ctx.fillText(`AUC: ${auc.toFixed(3)}`, w - 50, padT + 10);
  } else { ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center'; ctx.fillText('Train to see ROC', w / 2, h / 2 + 10); ctx.globalAlpha = 1; ctx.textAlign = 'start'; }
  ctx.fillStyle = muted; ctx.font = "9px 'Inter', sans-serif"; ctx.fillText('FPR', w / 2, h - 5);
  ctx.save(); ctx.translate(10, h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('TPR', -10, 0); ctx.restore();
}
