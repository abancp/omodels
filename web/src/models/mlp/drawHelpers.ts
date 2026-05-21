import type { Point, MLPState } from './math';

export const COLORS = {
  class0: '#a855f7',
  class1: '#4ade80',
  border: '#ffffff80',
  bg0: 'rgba(168, 85, 247, 0.2)',
  bg1: 'rgba(74, 222, 128, 0.2)',
};

export const FEATURE_COLORS = [
  '#ff4757', // Coral Red
  '#2ed573', // Bright Green
  '#1e90ff', // Neon Blue
  '#ffa502', // Orange
  '#eccc68', // Soft Gold
  '#a855f7', // Purple
  '#00d2d3', // Teal
];

export function drawDataCanvas(
  canvas: HTMLCanvasElement,
  points: Point[],
  state: MLPState,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  _dataset: string,
  inferResults: { features: number[]; pred: number[]; rawFeatures?: number[] }[],
  xAxisMode: 'standard' | 'multi-feature' = 'standard',
  selectedXFeatureIdx: number = 0
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

  if (xAxisMode === 'multi-feature') {
    // 1. Draw Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    // vertical grid lines
    for (let xVal = 0; xVal <= 1.05; xVal += 0.2) {
      const [px1, py1] = toPx(xVal, vp.yMin);
      const [px2, py2] = toPx(xVal, vp.yMax);
      ctx.beginPath(); ctx.moveTo(px1, py1); ctx.lineTo(px2, py2); ctx.stroke();
      // add x value labels at the bottom
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '9px monospace';
      ctx.fillText(xVal.toFixed(1), px1 - 8, H - 4);
    }
    // horizontal grid lines
    const yStep = (vp.yMax - vp.yMin) / 5 || 0.2;
    for (let yVal = vp.yMin; yVal <= vp.yMax; yVal += yStep) {
      const [px1, py1] = toPx(vp.xMin, yVal);
      const [px2, py2] = toPx(vp.xMax, yVal);
      ctx.beginPath(); ctx.moveTo(px1, py1); ctx.lineTo(px2, py2); ctx.stroke();
      // add y labels
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '9px monospace';
      ctx.fillText(yVal.toFixed(2), 4, py1 + 3);
    }

    // 2. Draw prediction curves (if model has been trained / has layers)
    if (state.layers && state.layers.length > 0) {
      const featuresToDraw = selectedXFeatureIdx === -1
        ? Array.from({ length: state.numInputs }, (_, idx) => idx)
        : [selectedXFeatureIdx];

      const meanFeatures = new Array(state.numInputs).fill(0.5);
      if (points.length > 0) {
        for (let f = 0; f < state.numInputs; f++) {
          let sum = 0;
          for (const p of points) sum += p.features[f] ?? 0.5;
          meanFeatures[f] = sum / points.length;
        }
      }

      ctx.shadowBlur = 0;
      for (const f of featuresToDraw) {
        ctx.strokeStyle = FEATURE_COLORS[f % FEATURE_COLORS.length];
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        
        const steps = 60;
        for (let s = 0; s <= steps; s++) {
          const vx = 0 + (s / steps) * 1;
          const features = [...meanFeatures];
          features[f] = vx;

          let currentIn = features;
          for (let l = 0; l < state.layers.length; l++) {
            const layer = state.layers[l];
            const nextActs = new Array(layer.nodes);
            for (let k = 0; k < layer.nodes; k++) {
              let sum = layer.biases[k] || 0;
              for (let fi = 0; fi < currentIn.length; fi++) {
                sum += currentIn[fi] * (layer.weights[k]?.[fi] || 0);
              }
              if (layer.activation === 'step') nextActs[k] = sum >= 0 ? 1 : 0;
              else if (layer.activation === 'sigmoid') nextActs[k] = 1 / (1 + Math.exp(-sum));
              else if (layer.activation === 'relu') nextActs[k] = Math.max(0, sum);
              else if (layer.activation === 'tanh') nextActs[k] = Math.tanh(sum);
              else nextActs[k] = sum;
            }
            currentIn = nextActs;
          }
          const predVal = currentIn.length > 1 ? (currentIn[1] || 0) : currentIn[0];

          const [px, py] = toPx(vx, predVal);
          if (s === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
    }

    // 3. Draw Points
    const featuresToDraw = selectedXFeatureIdx === -1
      ? Array.from({ length: state.numInputs }, (_, idx) => idx)
      : [selectedXFeatureIdx];

    for (const pt of points) {
      if (!pt.features) continue;
      for (const f of featuresToDraw) {
        const val = pt.features[f] ?? 0.5;
        const [px, py] = toPx(val, pt.label);
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fillStyle = FEATURE_COLORS[f % FEATURE_COLORS.length];
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.7)';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }

    // 4. Draw Inference Reference points
    for (const res of inferResults) {
      if (!res.features) continue;
      const predVal = res.pred.length > 1 ? (res.pred[1] || 0) : res.pred[0];
      for (const f of featuresToDraw) {
        const val = res.features[f] ?? 0.5;
        const [px, py] = toPx(val, predVal);
        
        ctx.beginPath();
        ctx.arc(px, py, 8, 0, Math.PI * 2);
        ctx.fillStyle = FEATURE_COLORS[f % FEATURE_COLORS.length];
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

  } else {
    // STANDARD DECISION SPACE GRID RENDERING (existing codebase logic)
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
        const predArray = currentIn;
        const predVal = predArray.length > 1 ? (predArray[1] || 0) : predArray[0];

        const [px, py] = toPx(vx, vy);
        const [px2, py2] = toPx(vx + dx, vy + dy);

        ctx.fillStyle = predVal > 0.5 ? COLORS.bg1 : COLORS.bg0;
        ctx.fillRect(px, Math.min(py, py2), Math.abs(px2 - px) + 1, Math.abs(py2 - py) + 1);
      }
    }
    ctx.globalAlpha = 1.0;

    // Draw Points
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

    // Inference Reference Points
    for (const res of inferResults) {
      if (!res.features || res.features.length < 1) continue;
      const [px, py] = toPx(res.features[0], state.numInputs > 1 ? res.features[1] : 0);
      const predVal = res.pred.length > 1 ? (res.pred[1] || 0) : res.pred[0];
      
      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = predVal > 0.5 ? COLORS.class1 : COLORS.class0;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 3;
      ctx.stroke();
      
      ctx.beginPath();
      ctx.arc(px, py, 16, 0, Math.PI * 2);
      ctx.strokeStyle = predVal > 0.5 ? COLORS.class1 : COLORS.class0;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}

export function drawNetworkCanvas(
  canvas: HTMLCanvasElement, 
  state: MLPState, 
  inferPt: Point | null, 
  inferActs: number[][] | null,
  inferPred: number[] | null
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
  const drawNode = (x: number, y: number, label: string, val: string, isOut = false, size = 14, isPredicted = false) => {
    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = isPredicted ? '#4ade80' : cSurfaceHighest; 
    ctx.fill();
    ctx.strokeStyle = isPredicted ? '#fff' : (isOut ? cPrimary : cBorder); 
    ctx.lineWidth = isPredicted ? 3 : (isOut ? 2 : 1); 
    ctx.stroke();
    
    if (isPredicted) {
      ctx.shadowColor = '#4ade80'; ctx.shadowBlur = 15;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    
    ctx.fillStyle = isPredicted ? '#111' : (isOut ? cPrimary : cOnSurfaceVariant); 
    ctx.font = isOut || isPredicted ? 'bold 12px Inter' : '10px Inter'; 
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
  let predictedIdx = -1;
  if (inferPred && inferPred.length > 0) {
    if (inferPred.length === 1) {
       predictedIdx = inferPred[0] >= 0.5 ? 0 : -1;
    } else {
       predictedIdx = inferPred.indexOf(Math.max(...inferPred));
    }
  }

  for (let l = 0; l < L; l++) {
    const layer = state.layers[l];
    const isOut = l === L - 1;
    for (let k = 0; k < layer.nodes; k++) {
      let val = '';
      if (inferActs && inferActs[l]) val = inferActs[l][k].toFixed(2);
      else if (isOut && inferPred) val = inferPred[k] !== undefined ? inferPred[k].toFixed(3) : 'out';
      else val = isOut ? `out ${k+1}` : `h${k+1}`;
      
      const label = isOut ? `y${layer.nodes > 1 ? k+1 : ''}` : 'Σ';
      const isPredicted = isOut && k === predictedIdx;
      drawNode(getX(l + 1), getY(k, layer.nodes), label, val, isOut, isOut ? 18 : 14, isPredicted);
    }
  }
}
