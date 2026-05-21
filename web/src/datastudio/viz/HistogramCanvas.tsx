/* Canvas-based Histogram for numeric columns */
import { useRef, useEffect } from 'react';
import type { NumericStats } from '../types';

interface Props {
  stats: NumericStats;
  width: number;
  height: number;
  accentColor?: string;
  label?: string;
}

export default function HistogramCanvas({ stats, width, height, accentColor = '#cfbcff', label }: Props) {
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

    const { histogram, min, max } = stats;
    if (!histogram || histogram.length === 0) return;
    const counts = histogram.map(h => h.count);

    const pad = { top: label ? 24 : 8, right: 20, bottom: 40, left: 60 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const maxCount = Math.max(...counts);

    // Label
    if (label) {
      ctx.font = '600 10px Inter';
      ctx.fillStyle = '#948e9c';
      ctx.textAlign = 'left';
      ctx.fillText(label.toUpperCase(), pad.left, 16);
    }

    // Bars
    const barW = plotW / counts.length - 1;
    for (let i = 0; i < counts.length; i++) {
      const barH = maxCount > 0 ? (counts[i] / maxCount) * plotH : 0;
      const x = pad.left + i * (barW + 1);
      const y = pad.top + plotH - barH;

      const alpha = 0.3 + 0.7 * (counts[i] / maxCount);
      ctx.fillStyle = accentColor + Math.round(alpha * 255).toString(16).padStart(2, '0');
      ctx.fillRect(x, y, barW, barH);
    }

    // X axis labels
    ctx.font = '400 10px JetBrains Mono';
    ctx.fillStyle = '#948e9c';
    ctx.textAlign = 'center';
    const formatNum = (n: number) => Math.abs(n) >= 1000 ? (n / 1000).toFixed(1) + 'k' : n.toFixed(1);
    ctx.fillText(formatNum(min), pad.left, height - 6);
    ctx.fillText(formatNum(max), width - pad.right, height - 6);

    // Y axis
    ctx.textAlign = 'right';
    ctx.fillText(String(maxCount), pad.left - 4, pad.top + 10);
    ctx.fillText('0', pad.left - 4, pad.top + plotH);

    // Axis lines
    ctx.strokeStyle = 'rgba(148,142,156,0.2)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();
  }, [stats, width, height, accentColor, label]);

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />;
}
