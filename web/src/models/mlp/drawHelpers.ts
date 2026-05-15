import type { Point, MLPState } from './math';

export const COLORS = {
  class0: '#a855f7',
  class1: '#4ade80',
  border: '#ffffff80',
  bg0: 'rgba(168, 85, 247, 0.2)',
  bg1: 'rgba(74, 222, 128, 0.2)',
};

export function drawDataCanvas(
  canvas: HTMLCanvasElement,
  points: Point[],
  state: MLPState,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  _dataset: string,
  inferResults: { features: number[]; pred: number }[]
) {
  const ctx = canvas.getContext('2d'); if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  ctx.clearRect(0, 0, W, H);

  if (!state.layers || state.layers.length === 0) return;

  const toPx = (x: number, y: number) => [
    ((x - vp.xMin) / (vp.xMax - vp.xMin)) * W,
    H - ((y - vp.yMin) / (vp.yMax - vp.yMin)) * H
  ];

  // Draw decision boundary grid
  const gridSize = 45;
  const dx = (vp.xMax - vp.xMin) / gridSize;
  const dy = (vp.yMax - vp.yMin) / gridSize;
  
  const baseFeatures = new Array(state.numInputs).fill(0.5);

  ctx.globalAlpha = 0.4;
  for (let i = 0; i < gridSize; i++) {
    for (let j = 0; j < gridSize; j++) {
      const vx = vp.xMin + i * dx;
      const vy = vp.yMin + j * dy;
      
      const features = [...baseFeatures];
      features[0] = vx;
      if (state.numInputs > 1) features[1] = vy;

      // Predict
      let currentIn = features;
      for (let l = 0; l < state.layers.length; l++) {
        const layer = state.layers[l];
        const nextActs = new Array(layer.nodes);
        for (let k = 0; k < layer.nodes; k++) {
          let sum = layer.biases[k] || 0;
          for (let f = 0; f < currentIn.length; f++) {
            sum += currentIn[f] * (layer.weights[k]?.[f] || 0);
          }
          if (layer.activation === 'step') nextActs[k] = sum >= 0 ? 1 : 0;
          else if (layer.activation === 'sigmoid') nextActs[k] = 1 / (1 + Math.exp(-sum));
          else if (layer.activation === 'relu') nextActs[k] = Math.max(0, sum);
          else if (layer.activation === 'tanh') nextActs[k] = Math.tanh(sum);
          else nextActs[k] = sum;
        }
        currentIn = nextActs;
      }
      const pred = currentIn[0]; // Output layer has 1 node

      const [px, py] = toPx(vx, vy);
      const [px2, py2] = toPx(vx + dx, vy + dy);

      ctx.fillStyle = pred > 0.5 ? COLORS.bg1 : COLORS.bg0;
      ctx.fillRect(px, Math.min(py, py2), Math.abs(px2 - px) + 1, Math.abs(py2 - py) + 1);
    }
  }
  ctx.globalAlpha = 1.0;

  // Draw points
  for (const pt of points) {
    if (!pt.features || pt.features.length < 1) continue;
    const [px, py] = toPx(pt.features[0], state.numInputs > 1 ? pt.features[1] : 0);
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = pt.label === 1 ? COLORS.class1 : COLORS.class0;
    ctx.fill();
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Inference pts
  for (const res of inferResults) {
    if (!res.features || res.features.length < 1) continue;
    const [px, py] = toPx(res.features[0], state.numInputs > 1 ? res.features[1] : 0);
    ctx.beginPath();
    ctx.arc(px, py, 10, 0, Math.PI * 2);
    ctx.fillStyle = res.pred > 0.5 ? COLORS.class1 : COLORS.class0;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    ctx.beginPath();
    ctx.arc(px, py, 16, 0, Math.PI * 2);
    ctx.strokeStyle = res.pred > 0.5 ? COLORS.class1 : COLORS.class0;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

export function drawNetworkCanvas(
  canvas: HTMLCanvasElement, 
  state: MLPState, 
  inferPt: Point | null, 
  inferActs: number[][] | null,
  inferPred: number | null
) {
  const ctx = canvas.getContext('2d'); if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  
  const root = getComputedStyle(document.documentElement);
  const cSurfaceHighest = root.getPropertyValue('--c-surface-container-highest').trim() || '#313244';

  const cPrimary = root.getPropertyValue('--c-primary').trim() || '#cba6f7';
  const cOnSurfaceVariant = root.getPropertyValue('--c-on-surface-variant').trim() || '#a6adc8';
  const cBorder = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.1)';
  
  ctx.clearRect(0, 0, W, H);

  if (!state.layers || state.layers.length === 0) return;

  const L = state.layers.length;
  // Layers total: Input + L hidden layers
  const totalLayers = L + 1;
  const spacingX = (W - 80) / (totalLayers - 1);
  const startX = 40;

  const getX = (layerIdx: number) => startX + layerIdx * spacingX;

  const getY = (nodeIdx: number, numNodes: number) => {
    const spacingY = Math.min(50, (H - 40) / numNodes);
    const startY = H / 2 - ((numNodes - 1) * spacingY) / 2;
    return startY + nodeIdx * spacingY;
  };

  const drawEdge = (sx: number, sy: number, ex: number, ey: number, weight: number) => {
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); 
    ctx.setLineDash(weight < 0 ? [3, 3] : []);
    ctx.strokeStyle = Math.abs(weight) > 1e-2 ? (weight > 0 ? '#4ade80' : '#f87171') : '#555';
    ctx.lineWidth = Math.max(0.5, Math.min(3, Math.abs(weight)));
    ctx.stroke();
    ctx.setLineDash([]);
  };

  // Edges
  for (let l = 0; l < L; l++) {
    const layer = state.layers[l];
    const prevNodes = l === 0 ? state.numInputs : state.layers[l - 1].nodes;
    const sx = getX(l);
    const ex = getX(l + 1);

    for (let k = 0; k < layer.nodes; k++) {
      const ey = getY(k, layer.nodes);
      for (let j = 0; j < prevNodes; j++) {
        const sy = getY(j, prevNodes);
        drawEdge(sx, sy, ex, ey, layer.weights[k]?.[j] || 0);
      }
    }
  }

  // Draw Nodes
  const drawNode = (x: number, y: number, label: string, val: string, isOut = false, size = 14) => {
    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = cSurfaceHighest; 
    ctx.fill();
    ctx.strokeStyle = isOut ? cPrimary : cBorder; ctx.lineWidth = isOut ? 2 : 1; ctx.stroke();
    
    ctx.fillStyle = isOut ? cPrimary : cOnSurfaceVariant; ctx.font = isOut ? 'bold 12px Inter' : '10px Inter'; 
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y);
    
    if (val) {
      ctx.fillStyle = cOnSurfaceVariant; ctx.font = '9px monospace';
      ctx.fillText(val, x, y + size + 10);
    }
  };

  // Inputs
  for (let j = 0; j < state.numInputs; j++) {
    const val = inferPt?.features ? inferPt.features[j].toFixed(2) : `x${j+1}`;
    drawNode(getX(0), getY(j, state.numInputs), `x${j+1}`, val, false, 12);
  }

  // Hidden Layers and Output
  for (let l = 0; l < L; l++) {
    const layer = state.layers[l];
    const isOut = l === L - 1;
    for (let k = 0; k < layer.nodes; k++) {
      let val = '';
      if (inferActs && inferActs[l]) val = inferActs[l][k].toFixed(2);
      else if (isOut) val = inferPred !== null ? inferPred.toFixed(3) : 'out';
      else val = `h${k+1}`;
      
      const label = isOut ? 'y' : 'Σ';
      drawNode(getX(l + 1), getY(k, layer.nodes), label, val, isOut, isOut ? 18 : 14);
    }
  }
}
