/**
 * Canvas drawing helpers for K-Means visualization.
 */
import { type KMeansState, type Point, predictCluster, CLUSTER_COLORS } from './math';

/* ─── Draw cluster map with Voronoi ─── */
export function drawDataCanvas(
  canvas: HTMLCanvasElement, points: Point[], state: KMeansState | null,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  inferResults: { x: number; y: number; cluster: number }[], dataset: string,
  showVoronoi: boolean, showCentroidPath: boolean,
  testPoints?: Point[]
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

  // Voronoi regions
  if (state && state.centroids.length > 1 && showVoronoi) {
    const res = 6;
    for (let px = 0; px < W; px += res) {
      for (let py = 0; py < H; py += res) {
        const nx = vp.xMin + ((px + res / 2) / W) * (vp.xMax - vp.xMin);
        const ny = vp.yMax - ((py + res / 2) / H) * (vp.yMax - vp.yMin);
        const c = predictCluster(nx, ny, state.centroids);
        ctx.fillStyle = `${CLUSTER_COLORS[c % CLUSTER_COLORS.length]}12`;
        ctx.fillRect(px, py, res, res);
      }
    }
  }

  // Grid
  ctx.strokeStyle = gridColor; ctx.lineWidth = 1; ctx.beginPath();
  const getTicks = (min: number, max: number) => { const step = Math.pow(10, Math.floor(Math.log10((max - min) / 5))); const t = []; for (let v = Math.ceil(min / step) * step; v <= max; v += step) t.push(v); return t; };
  for (const t of getTicks(vp.xMin, vp.xMax)) { const px = mapX(t); ctx.moveTo(px, 0); ctx.lineTo(px, H); }
  for (const t of getTicks(vp.yMin, vp.yMax)) { const py = mapY(t); ctx.moveTo(0, py); ctx.lineTo(W, py); }
  ctx.stroke();
  ctx.fillStyle = textColor; ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = 'top'; for (const t of getTicks(vp.xMin, vp.xMax)) ctx.fillText(t.toFixed(1), mapX(t) + 4, H - 16);
  ctx.textBaseline = 'bottom'; for (const t of getTicks(vp.yMin, vp.yMax)) ctx.fillText(t.toFixed(1), 4, mapY(t) - 4);

  // Points colored by cluster
  if (state && state.assignments.length === points.length) {
    for (let i = 0; i < points.length; i++) {
      const p = points[i]; const c = state.assignments[i];
      ctx.beginPath(); ctx.arc(mapX(p.x), mapY(p.y), 4, 0, Math.PI * 2);
      ctx.fillStyle = CLUSTER_COLORS[c % CLUSTER_COLORS.length]; ctx.fill();
      ctx.strokeStyle = '#00000030'; ctx.lineWidth = 0.5; ctx.stroke();
    }
  } else {
    for (const p of points) {
      ctx.beginPath(); ctx.arc(mapX(p.x), mapY(p.y), 4, 0, Math.PI * 2);
      ctx.fillStyle = '#888888'; ctx.fill();
    }
  }

  // Draw test points if present
  if (testPoints && testPoints.length > 0) {
    for (const p of testPoints) {
      const sx = mapX(p.x), sy = mapY(p.y);
      ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#10b981'; ctx.fill();
      ctx.strokeStyle = '#00000040'; ctx.lineWidth = 0.5; ctx.stroke();
      ctx.beginPath(); ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#10b98140'; ctx.lineWidth = 0.5; ctx.stroke();
    }
  }

  // Centroid path history
  if (state && showCentroidPath && state.centroidHistory.length > 1) {
    for (let c = 0; c < state.k; c++) {
      ctx.beginPath();
      ctx.strokeStyle = `${CLUSTER_COLORS[c % CLUSTER_COLORS.length]}60`;
      ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      for (let it = 0; it < state.centroidHistory.length; it++) {
        const cent = state.centroidHistory[it][c];
        if (!cent) continue;
        if (it === 0) ctx.moveTo(mapX(cent.x), mapY(cent.y));
        else ctx.lineTo(mapX(cent.x), mapY(cent.y));
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
  }

  // Centroids (big markers)
  if (state) {
    for (let c = 0; c < state.centroids.length; c++) {
      const cent = state.centroids[c]; const color = CLUSTER_COLORS[c % CLUSTER_COLORS.length];
      const cx = mapX(cent.x), cy = mapY(cent.y);
      // Outer glow
      ctx.beginPath(); ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.fillStyle = `${color}30`; ctx.fill();
      // Diamond shape
      ctx.beginPath(); ctx.moveTo(cx, cy - 8); ctx.lineTo(cx + 6, cy); ctx.lineTo(cx, cy + 8); ctx.lineTo(cx - 6, cy); ctx.closePath();
      ctx.fillStyle = color; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      // Label
      ctx.fillStyle = '#000'; ctx.font = "bold 8px 'Inter', sans-serif"; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(c + 1), cx, cy); ctx.textAlign = 'start';
    }
  }

  // Inference markers
  for (const ir of inferResults) {
    const sx = mapX(ir.x), sy = mapY(ir.y);
    ctx.beginPath(); ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fillStyle = CLUSTER_COLORS[ir.cluster % CLUSTER_COLORS.length];
    ctx.globalAlpha = 0.8; ctx.fill(); ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(sx, sy, 10, 0, Math.PI * 2);
    ctx.strokeStyle = `${CLUSTER_COLORS[ir.cluster % CLUSTER_COLORS.length]}60`; ctx.lineWidth = 1.5; ctx.stroke();
  }

  if (dataset === 'custom' && points.length < 3) {
    ctx.fillStyle = textColor; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.5; ctx.textAlign = 'center';
    ctx.fillText('Click to add data points', W / 2, H / 2);
    ctx.globalAlpha = 1; ctx.textAlign = 'start';
  }
}

/* ─── Draw Inertia Curve ─── */
export function drawInertiaCanvas(canvas: HTMLCanvasElement, state: KMeansState | null) {
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

  ctx.fillStyle = muted; ctx.font = "600 10px 'Inter', sans-serif"; ctx.globalAlpha = 0.6; ctx.fillText('INERTIA (WCSS) PER ITERATION', 12, 18); ctx.globalAlpha = 1;

  if (!state || state.inertiaHistory.length < 2) {
    ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
    ctx.fillText('Train to see inertia curve', w / 2, h / 2 + 10); ctx.globalAlpha = 1; ctx.textAlign = 'start'; return;
  }

  const padL = 45, padR = 12, padT = 30, padB = 24;
  const cw = w - padL - padR, ch = h - padT - padB;
  ctx.strokeStyle = border; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

  const hist = state.inertiaHistory;
  const maxV = Math.max(...hist) * 1.1 || 1;

  ctx.beginPath();
  for (let i = 0; i < hist.length; i++) {
    const x = padL + (i / (hist.length - 1)) * cw;
    const y = padT + ((maxV - hist[i]) / maxV) * ch;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = primary; ctx.lineWidth = 1.5; ctx.stroke();
  // Fill area
  const lastX = padL + ((hist.length - 1) / (hist.length - 1)) * cw;
  const gradient = ctx.createLinearGradient(0, padT, 0, h - padB);
  gradient.addColorStop(0, `${primary}20`); gradient.addColorStop(1, `${primary}02`);
  ctx.lineTo(lastX, h - padB); ctx.lineTo(padL, h - padB); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();

  ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
  ctx.fillText('0', padL - 6, h - padB + 12); ctx.fillText(String(hist.length - 1), w - padR - 10, h - padB + 12);
  ctx.fillText(maxV.toFixed(2), 2, padT + 4); ctx.globalAlpha = 1;
}

/* ─── Draw Elbow Plot ─── */
export function drawElbowCanvas(canvas: HTMLCanvasElement, elbowData: { k: number; inertia: number }[] | null) {
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

  if (!elbowData || elbowData.length < 2) {
    ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
    ctx.fillText('Train to see elbow plot', w / 2, h / 2 + 10); ctx.globalAlpha = 1; ctx.textAlign = 'start'; return;
  }

  const padL = 40, padR = 12, padT = 10, padB = 24;
  const cw = w - padL - padR, ch = h - padT - padB;
  ctx.strokeStyle = border; ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();

  const maxI = Math.max(...elbowData.map(d => d.inertia)) * 1.1 || 1;
  const maxK = elbowData[elbowData.length - 1].k;
  const minK = elbowData[0].k;

  ctx.beginPath();
  for (let i = 0; i < elbowData.length; i++) {
    const x = padL + ((elbowData[i].k - minK) / (maxK - minK)) * cw;
    const y = padT + ((maxI - elbowData[i].inertia) / maxI) * ch;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    // Point marker
    ctx.fillStyle = primary;
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.strokeStyle = primary; ctx.lineWidth = 1.5; ctx.stroke();

  ctx.fillStyle = muted; ctx.font = "9px 'JetBrains Mono', monospace"; ctx.globalAlpha = 0.5;
  for (const d of elbowData) {
    const x = padL + ((d.k - minK) / (maxK - minK)) * cw;
    ctx.fillText(`k=${d.k}`, x - 8, h - 8);
  }
  ctx.globalAlpha = 1;

  ctx.fillStyle = muted; ctx.font = "9px 'Inter', sans-serif";
  ctx.fillText('K', w / 2, h - 2);
}
