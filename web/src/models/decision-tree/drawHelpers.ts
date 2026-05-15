/**
 * Canvas drawing helpers for Decision Tree visualization.
 */
import { type DecisionTreeState, type Point, predictSingle, predictProbability, layoutTree, computeROCCurve } from './math';

/* ─── Draw Decision Boundary + Points ─── */
export function drawDataCanvas(
  canvas: HTMLCanvasElement,
  points: Point[],
  state: DecisionTreeState | null,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  inferResults: { x: number; y: number; cls: number }[],
  dataset: string
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
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

  // Decision boundary background
  if (state?.root) {
    const res = 6;
    for (let px = 0; px < W; px += res) {
      for (let py = 0; py < H; py += res) {
        const nx = vp.xMin + ((px + res / 2) / W) * (vp.xMax - vp.xMin);
        const ny = vp.yMax - ((py + res / 2) / H) * (vp.yMax - vp.yMin);
        const pred = predictSingle(nx, ny, state.root);
        ctx.fillStyle = pred === 1 ? `${tertiary}15` : `${primary}15`;
        ctx.fillRect(px, py, res, res);
      }
    }
    // Boundary contour
    for (let px = 0; px < W; px += 3) {
      for (let py = 0; py < H; py += 3) {
        const nx = vp.xMin + (px / W) * (vp.xMax - vp.xMin);
        const ny = vp.yMax - (py / H) * (vp.yMax - vp.yMin);
        const prob = predictProbability(nx, ny, state.root);
        if (Math.abs(prob - 0.5) < 0.02) {
          ctx.fillStyle = '#ffffff50';
          ctx.fillRect(px, py, 3, 3);
        }
      }
    }
    // Draw split lines
    drawSplitLines(ctx, state.root, vp, W, H, mapX, mapY);
  }

  // Grid
  ctx.strokeStyle = gridColor; ctx.lineWidth = 1; ctx.beginPath();
  const getTicks = (min: number, max: number) => {
    const range = max - min;
    const step = Math.pow(10, Math.floor(Math.log10(range / 5)));
    const ticks = [];
    for (let t = Math.ceil(min / step) * step; t <= max; t += step) ticks.push(t);
    return ticks;
  };
  for (const t of getTicks(vp.xMin, vp.xMax)) { const px = mapX(t); ctx.moveTo(px, 0); ctx.lineTo(px, H); }
  for (const t of getTicks(vp.yMin, vp.yMax)) { const py = mapY(t); ctx.moveTo(0, py); ctx.lineTo(W, py); }
  ctx.stroke();

  ctx.fillStyle = textColor; ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = 'top';
  for (const t of getTicks(vp.xMin, vp.xMax)) ctx.fillText(t.toFixed(1), mapX(t) + 4, H - 16);
  ctx.textBaseline = 'bottom';
  for (const t of getTicks(vp.yMin, vp.yMax)) ctx.fillText(t.toFixed(1), 4, mapY(t) - 4);

  // Points
  for (const p of points) {
    ctx.beginPath(); ctx.arc(mapX(p.x), mapY(p.y), 5, 0, Math.PI * 2);
    ctx.fillStyle = p.cls === 0 ? primary : tertiary; ctx.fill();
    ctx.strokeStyle = '#00000040'; ctx.lineWidth = 1; ctx.stroke();
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

  if (dataset === 'custom' && points.length < 3) {
    ctx.fillStyle = textColor; ctx.font = "11px 'Inter', sans-serif";
    ctx.globalAlpha = 0.5; ctx.textAlign = 'center';
    ctx.fillText('Click to add points (Shift+Click for class 1)', W / 2, H / 2);
    ctx.globalAlpha = 1; ctx.textAlign = 'start';
  }
}

function drawSplitLines(
  ctx: CanvasRenderingContext2D,
  node: { isLeaf: boolean; splitFeature?: 'x' | 'y'; splitValue?: number; left?: any; right?: any; depth: number },
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  W: number, H: number,
  mapX: (x: number) => number,
  mapY: (y: number) => number,
  xBounds = { min: vp.xMin, max: vp.xMax },
  yBounds = { min: vp.yMin, max: vp.yMax }
) {
  if (node.isLeaf || !node.splitFeature || node.splitValue === undefined) return;
  const alpha = Math.max(0.15, 0.6 - node.depth * 0.1);
  ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
  ctx.lineWidth = Math.max(0.5, 2 - node.depth * 0.3);
  ctx.setLineDash([4, 3]);

  if (node.splitFeature === 'x') {
    const px = mapX(node.splitValue);
    const y1 = mapY(yBounds.max), y2 = mapY(yBounds.min);
    ctx.beginPath(); ctx.moveTo(px, y1); ctx.lineTo(px, y2); ctx.stroke();
    if (node.left) drawSplitLines(ctx, node.left, vp, W, H, mapX, mapY, { min: xBounds.min, max: node.splitValue }, yBounds);
    if (node.right) drawSplitLines(ctx, node.right, vp, W, H, mapX, mapY, { min: node.splitValue, max: xBounds.max }, yBounds);
  } else {
    const py = mapY(node.splitValue);
    const x1 = mapX(xBounds.min), x2 = mapX(xBounds.max);
    ctx.beginPath(); ctx.moveTo(x1, py); ctx.lineTo(x2, py); ctx.stroke();
    if (node.left) drawSplitLines(ctx, node.left, vp, W, H, mapX, mapY, xBounds, { min: yBounds.min, max: node.splitValue });
    if (node.right) drawSplitLines(ctx, node.right, vp, W, H, mapX, mapY, xBounds, { min: node.splitValue, max: yBounds.max });
  }
  ctx.setLineDash([]);
}

/* ─── Draw Tree Structure ─── */
export function drawTreeCanvas(canvas: HTMLCanvasElement, state: DecisionTreeState | null) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const root = getComputedStyle(document.documentElement);
  const primary = root.getPropertyValue('--c-primary').trim() || '#cfbcff';
  const tertiary = root.getPropertyValue('--c-tertiary').trim() || '#e7c365';
  const muted = root.getPropertyValue('--c-on-surface-variant').trim() || '#cbc4d2';
  const border = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';

  ctx.fillStyle = muted; ctx.font = "600 10px 'Inter', sans-serif";
  ctx.globalAlpha = 0.6; ctx.fillText('TREE STRUCTURE', 12, 18); ctx.globalAlpha = 1;

  if (!state?.root) {
    ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
    ctx.fillText('Train to see decision tree', W / 2, H / 2 + 10);
    ctx.globalAlpha = 1; ctx.textAlign = 'start';
    return;
  }

  const layout = layoutTree(state.root);
  if (layout.length === 0) return;

  const padL = 30, padR = 30, padT = 35, padB = 20;
  const cw = W - padL - padR, ch = H - padT - padB;
  const maxX = Math.max(...layout.map(n => n.x));
  const maxY = Math.max(...layout.map(n => n.y));
  const nodeR = Math.min(14, Math.max(6, cw / (maxX + 2) / 3));

  const mapNX = (x: number) => padL + (maxX > 0 ? (x / maxX) * cw : cw / 2);
  const mapNY = (y: number) => padT + (maxY > 0 ? (y / maxY) * ch : 0);

  // Edges
  for (const node of layout) {
    if (node.parentX !== undefined && node.parentY !== undefined) {
      ctx.beginPath();
      ctx.moveTo(mapNX(node.parentX), mapNY(node.parentY) + nodeR);
      ctx.lineTo(mapNX(node.x), mapNY(node.y) - nodeR);
      ctx.strokeStyle = border; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }

  // Nodes
  for (const node of layout) {
    const nx = mapNX(node.x), ny = mapNY(node.y);
    ctx.beginPath();
    if (node.isLeaf) {
      ctx.rect(nx - nodeR, ny - nodeR, nodeR * 2, nodeR * 2);
    } else {
      ctx.arc(nx, ny, nodeR, 0, Math.PI * 2);
    }
    ctx.fillStyle = node.predictedClass === 1 ? `${tertiary}40` : `${primary}40`;
    ctx.fill();
    ctx.strokeStyle = node.predictedClass === 1 ? tertiary : primary;
    ctx.lineWidth = 1.5; ctx.stroke();

    // Label
    if (nodeR >= 8) {
      ctx.fillStyle = muted; ctx.font = `${Math.min(9, nodeR)}px 'JetBrains Mono', monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      if (node.isLeaf) {
        ctx.fillText(`C${node.predictedClass}`, nx, ny);
      } else {
        ctx.fillText(`${node.splitFeature}`, nx, ny - 3);
        ctx.font = `${Math.min(7, nodeR - 2)}px 'JetBrains Mono', monospace`;
        ctx.fillText(`${node.splitValue?.toFixed(2)}`, nx, ny + 5);
      }
      ctx.textAlign = 'start';
    }
  }
}

/* ─── Draw ROC Curve ─── */
export function drawROCCanvas(canvas: HTMLCanvasElement, points: Point[], state: DecisionTreeState | null) {
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
  const bdr = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.08)';

  const padL = 30, padB = 20, padT = 10, padR = 10;
  const cw = w - padL - padR, ch = h - padT - padB;

  ctx.strokeStyle = bdr; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, h - padB); ctx.lineTo(w - padR, h - padB); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(padL, h - padB); ctx.lineTo(w - padR, padT); ctx.setLineDash([3, 3]); ctx.stroke(); ctx.setLineDash([]);

  if (state?.root) {
    const { curve, auc } = computeROCCurve(points, state);
    if (curve.length > 0) {
      ctx.beginPath();
      for (let i = 0; i < curve.length; i++) {
        const x = padL + curve[i].fpr * cw, y = h - padB - curve[i].tpr * ch;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = primary; ctx.lineWidth = 2; ctx.stroke();
    }
    ctx.fillStyle = primary; ctx.font = "600 10px 'Inter', sans-serif";
    ctx.fillText(`AUC: ${auc.toFixed(3)}`, w - 50, padT + 10);
  } else {
    ctx.fillStyle = muted; ctx.font = "11px 'Inter', sans-serif"; ctx.globalAlpha = 0.3; ctx.textAlign = 'center';
    ctx.fillText('Train to see ROC', w / 2, h / 2 + 10);
    ctx.globalAlpha = 1; ctx.textAlign = 'start';
  }

  ctx.fillStyle = muted; ctx.font = "9px 'Inter', sans-serif";
  ctx.fillText('FPR', w / 2, h - 5);
  ctx.save(); ctx.translate(10, h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('TPR', -10, 0); ctx.restore();
}
