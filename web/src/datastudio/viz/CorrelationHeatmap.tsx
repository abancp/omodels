/* Canvas-based Correlation Heatmap */
import { useRef, useEffect } from 'react';

interface Props {
  cols: string[];
  matrix: number[][];
  width: number;
  height: number;
}

export default function CorrelationHeatmap({ cols, matrix, width, height }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const n = cols.length;
    if (n === 0) return;

    const labelPad = 60;
    const cellSize = Math.min((width - labelPad) / n, (height - labelPad) / n);
    const offsetX = labelPad;
    const offsetY = labelPad;

    // Color interpolation: negative = blue, zero = black, positive = purple
    function corrColor(r: number): string {
      const clamped = Math.max(-1, Math.min(1, r));
      if (clamped >= 0) {
        const t = clamped;
        const red = Math.round(207 * t);
        const green = Math.round(188 * t);
        const blue = Math.round(255 * t);
        return `rgb(${red},${green},${blue})`;
      } else {
        const t = -clamped;
        const red = Math.round(128 * t);
        const green = Math.round(180 * t);
        const blue = Math.round(255 * t);
        return `rgb(${red},${green},${blue})`;
      }
    }

    // Draw cells
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = offsetX + j * cellSize;
        const y = offsetY + i * cellSize;
        ctx.fillStyle = corrColor(matrix[i][j]);
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

        // Value text
        if (cellSize > 24) {
          ctx.font = '400 9px JetBrains Mono';
          ctx.fillStyle = Math.abs(matrix[i][j]) > 0.5 ? '#0f0d13' : '#948e9c';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(matrix[i][j].toFixed(2), x + cellSize / 2, y + cellSize / 2);
        }
      }
    }

    // Row labels
    ctx.font = '400 9px JetBrains Mono';
    ctx.fillStyle = '#cbc4d2';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      const label = cols[i].length > 8 ? cols[i].slice(0, 7) + '…' : cols[i];
      ctx.fillText(label, offsetX - 4, offsetY + i * cellSize + cellSize / 2);
    }

    // Column labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    for (let j = 0; j < n; j++) {
      ctx.save();
      const x = offsetX + j * cellSize + cellSize / 2;
      const y = offsetY - 4;
      ctx.translate(x, y);
      ctx.rotate(-Math.PI / 4);
      const label = cols[j].length > 8 ? cols[j].slice(0, 7) + '…' : cols[j];
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }
  }, [cols, matrix, width, height]);

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />;
}
