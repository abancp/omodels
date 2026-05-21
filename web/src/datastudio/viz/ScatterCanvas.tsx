/* Canvas-based Scatter Plot — Enhanced */
import { useRef, useEffect } from 'react';

interface Props {
  xData: number[];
  yData: number[];
  xLabel: string;
  yLabel: string;
  width: number;
  height: number;
  colorData?: any[];
  sizeData?: number[];
  options?: {
    pointSize?: number;
    opacity?: number;
    colorScheme?: 'default' | 'vibrant' | 'mono';
  };
}

const PALETTES = {
  default: ['#cfbcff', '#e7c365', '#ff8a80', '#80cbc4', '#b39ddb', '#ffab91', '#81d4fa'],
  vibrant: ['#ff00ff', '#00ffff', '#ffff00', '#00ff00', '#ff0000', '#0000ff', '#ffa500'],
  mono: ['#ffffff', '#cccccc', '#999999', '#666666', '#333333', '#aaaaaa', '#777777'],
};

export default function ScatterCanvas({ 
  xData, yData, xLabel, yLabel, width, height, colorData, sizeData, options 
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { pointSize = 2.5, opacity = 0.6, colorScheme = 'default' } = options || {};

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

    if (xData.length === 0 || yData.length === 0) return;

    const pad = { top: 12, right: 24, bottom: 44, left: 64 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const xMin = Math.min(...xData), xMax = Math.max(...xData);
    const yMin = Math.min(...yData), yMax = Math.max(...yData);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const palette = PALETTES[colorScheme];

    // Axis Grid
    ctx.strokeStyle = 'rgba(148,142,156,0.1)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (i / 4) * plotH;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
      const x = pad.left + (i / 4) * plotW;
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
    }

    // Points
    const n = Math.min(xData.length, yData.length, 5000);
    const opHex = Math.round(opacity * 255).toString(16).padStart(2, '0');

    for (let i = 0; i < n; i++) {
      const px = pad.left + ((xData[i] - xMin) / xRange) * plotW;
      const py = pad.top + plotH - ((yData[i] - yMin) / yRange) * plotH;

      let color = palette[0];
      if (colorData) {
        const val = colorData[i];
        const idx = typeof val === 'number' ? Math.abs(Math.round(val)) : String(val).length;
        color = palette[idx % palette.length];
      }

      let rad = pointSize;
      if (sizeData) {
        const sMin = Math.min(...sizeData), sMax = Math.max(...sizeData);
        const sRange = sMax - sMin || 1;
        rad = 1 + ((sizeData[i] - sMin) / sRange) * (pointSize * 2);
      }

      ctx.fillStyle = color + opHex;
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fill();
    }

    // Labels
    const fmt = (n: number) => Math.abs(n) >= 1000 ? (n / 1000).toFixed(1) + 'k' : n.toFixed(1);
    ctx.font = '500 10px JetBrains Mono';
    ctx.fillStyle = '#948e9c';
    ctx.textAlign = 'center';
    ctx.fillText(fmt(xMin), pad.left, height - 20);
    ctx.fillText(fmt(xMax), width - pad.right, height - 20);
    ctx.font = '600 11px Inter';
    ctx.fillText(xLabel.toUpperCase(), pad.left + plotW / 2, height - 10);

    ctx.textAlign = 'right';
    ctx.font = '500 10px JetBrains Mono';
    ctx.fillText(fmt(yMin), pad.left - 6, pad.top + plotH);
    ctx.fillText(fmt(yMax), pad.left - 6, pad.top + 6);

    ctx.save();
    ctx.translate(12, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.font = '600 11px Inter';
    ctx.fillText(yLabel.toUpperCase(), 0, 0);
    ctx.restore();

    // Border
    ctx.strokeStyle = 'rgba(148,142,156,0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, plotW, plotH);
  }, [xData, yData, xLabel, yLabel, width, height, colorData, sizeData, options]);

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />;
}
