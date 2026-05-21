import type { Point, PerceptronState } from './math';

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
  state: PerceptronState,
  vp: { xMin: number; xMax: number; yMin: number; yMax: number },
  _dataset: string,
  inferResults: { features: number[]; pred: number; rawFeatures?: number[] }[],
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

  // Defensive check for state migration/HMR
  if (!state.hiddenWeights || !state.hiddenBias || !state.outWeights) return;

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

    // 2. Draw prediction curves (if model has been trained / has weights)
    if (state.hiddenWeights && state.hiddenWeights.length > 0) {
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

          // Run prediction
          const hiddenActs = new Array(state.numPerceptrons);
          for (let k = 0; k < state.numPerceptrons; k++) {
            let sum = state.hiddenBias[k] || 0;
            for (let fi = 0; fi < state.numInputs; fi++) {
              sum += features[fi] * (state.hiddenWeights[k]?.[fi] || 0);
            }
            let act = sum;
            if (state.activation === 'step') act = sum >= 0 ? 1 : 0;
            else if (state.activation === 'sigmoid') act = 1 / (1 + Math.exp(-sum));
            else if (state.activation === 'relu') act = Math.max(0, sum);
            else if (state.activation === 'tanh') act = Math.tanh(sum);
            hiddenActs[k] = act;
          }
          let outSum = state.outBias || 0;
          for (let k = 0; k < state.numPerceptrons; k++) {
            outSum += hiddenActs[k] * (state.outWeights[k] || 0);
          }
          const pred = 1 / (1 + Math.exp(-outSum));

          const [px, py] = toPx(vx, pred);
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
      const [px, py] = toPx(res.features[selectedXFeatureIdx === -1 ? 0 : selectedXFeatureIdx], res.pred);
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = selectedXFeatureIdx === -1 ? '#fff' : FEATURE_COLORS[selectedXFeatureIdx % FEATURE_COLORS.length];
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

  } else {
    // STANDARD DECISION SPACE GRID RENDERING (existing codebase logic)
    const gridSize = 40;
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
        const hiddenActs = new Array(state.numPerceptrons);
        for (let k = 0; k < state.numPerceptrons; k++) {
          let sum = state.hiddenBias[k] || 0;
          for (let f = 0; f < state.numInputs; f++) {
            sum += features[f] * (state.hiddenWeights[k]?.[f] || 0);
          }
          let act = sum;
          if (state.activation === 'step') act = sum >= 0 ? 1 : 0;
          else if (state.activation === 'sigmoid') act = 1 / (1 + Math.exp(-sum));
          else if (state.activation === 'relu') act = Math.max(0, sum);
          else if (state.activation === 'tanh') act = Math.tanh(sum);
          hiddenActs[k] = act;
        }
        let outSum = state.outBias || 0;
        for (let k = 0; k < state.numPerceptrons; k++) {
          outSum += hiddenActs[k] * (state.outWeights[k] || 0);
        }
        const pred = 1 / (1 + Math.exp(-outSum));

        const [px, py] = toPx(vx, vy);
        const [px2, py2] = toPx(vx + dx, vy + dy);

        ctx.fillStyle = pred > 0.5 ? COLORS.bg1 : COLORS.bg0;
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
}

export function drawNetworkCanvas(
  canvas: HTMLCanvasElement, 
  state: PerceptronState, 
  inferPt: Point | null, 
  inferPred: { hiddenActs: number[], pred: number } | null
) {
  const ctx = canvas.getContext('2d'); if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  
  const root = getComputedStyle(document.documentElement);
  const cSurfaceHighest = root.getPropertyValue('--c-surface-container-highest').trim() || '#313244';
  const cSurfaceVariant = root.getPropertyValue('--c-surface-variant').trim() || '#45475a';
  const cPrimary = root.getPropertyValue('--c-primary').trim() || '#cba6f7';
  const cOnSurfaceVariant = root.getPropertyValue('--c-on-surface-variant').trim() || '#a6adc8';
  const cBorder = root.getPropertyValue('--c-panel-border').trim() || 'rgba(255,255,255,0.1)';
  
  ctx.clearRect(0, 0, W, H);

  // Defensive check
  if (!state.hiddenWeights || !state.hiddenBias || !state.outWeights) return;

  const numInputs = state.numInputs;
  const numHidden = state.numPerceptrons;
  
  const inX = Math.max(60, W * 0.15);
  const midX = W / 2;
  const outX = Math.min(W - 60, W * 0.85);
  
  const inYSpacing = Math.min(60, (H - 40) / numInputs);
  const inStartY = H / 2 - ((numInputs - 1) * inYSpacing) / 2;
  
  const hidYSpacing = Math.min(70, (H - 40) / numHidden);
  const hidStartY = H / 2 - ((numHidden - 1) * hidYSpacing) / 2;

  const getInY = (i: number) => inStartY + i * inYSpacing;
  const getHidY = (i: number) => hidStartY + i * hidYSpacing;
  
  // Draw edges
  const drawEdge = (sx: number, sy: number, ex: number, ey: number, weight: number) => {
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(ex, ey); 
    ctx.setLineDash(weight < 0 ? [5, 5] : []);
    ctx.strokeStyle = Math.abs(weight) > 1e-2 ? (weight > 0 ? '#4ade80' : '#f87171') : '#555';
    ctx.lineWidth = Math.max(1.0, Math.min(4, Math.abs(weight) * 1.5));
    ctx.stroke();
    ctx.setLineDash([]);
  };

  // 1. Edges: Input -> Hidden
  for (let k = 0; k < numHidden; k++) {
    for (let j = 0; j < numInputs; j++) {
      const weight = state.hiddenWeights[k]?.[j] || 0;
      drawEdge(inX, getInY(j), midX, getHidY(k), weight);
    }
  }

  // 2. Edges: Hidden -> Output
  for (let k = 0; k < numHidden; k++) {
    const weight = state.outWeights[k] || 0;
    drawEdge(midX, getHidY(k), outX, H/2, weight);
  }

  // 3. Draw Nodes
  const drawNode = (x: number, y: number, label: string, val: string, isConst = false, textAbove = false, size = 18) => {
    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fillStyle = isConst ? cSurfaceVariant : cSurfaceHighest; 
    ctx.fill();
    ctx.strokeStyle = isConst ? cBorder : cPrimary; ctx.lineWidth = isConst ? 1.5 : 2.5; ctx.stroke();
    
    ctx.fillStyle = isConst ? cOnSurfaceVariant : cPrimary; ctx.font = isConst ? '13px Inter' : 'bold 16px Inter'; 
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(label, x, y);
    
    // Value text
    ctx.fillStyle = cOnSurfaceVariant; ctx.font = '11px monospace';
    ctx.fillText(val, x, textAbove ? y - size - 10 : y + size + 16);
  };

  // Input nodes
  for (let j = 0; j < numInputs; j++) {
    const valStr = (inferPt && inferPt.features) ? inferPt.features[j]?.toFixed(2) : `in ${j+1}`;
    drawNode(inX, getInY(j), `x${j+1}`, valStr, true);
  }

  // Hidden nodes
  for (let k = 0; k < numHidden; k++) {
    const valStr = (inferPred && inferPred.hiddenActs) ? inferPred.hiddenActs[k]?.toFixed(2) : `h${k+1}`;
    drawNode(midX, getHidY(k), 'Σ', valStr, false, false, 24);
    
    // draw bias inside or near hidden node
    ctx.fillStyle = '#fff'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
    const bias = state.hiddenBias[k] || 0;
    ctx.fillText(`b=${bias.toFixed(2)}`, midX, getHidY(k) - 34);
  }

  // Output node
  const outVal = inferPred ? inferPred.pred.toFixed(3) : 'out';
  drawNode(outX, H/2, 'y', outVal, false, false, 24);
  ctx.fillStyle = '#fff'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  const outBias = state.outBias || 0;
  ctx.fillText(`b=${outBias.toFixed(2)}`, outX, H/2 - 34);
}
