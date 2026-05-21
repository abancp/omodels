/* Canvas-based Box Plot */
import { useRef, useEffect } from 'react';
import type { NumericStats } from '../types';

interface Props {
  stats: NumericStats[];
  labels: string[];
  width: number;
  height: number;
}

export default function BoxPlotCanvas({ stats, labels, width, height }: Props) {
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

    const n = stats.length;
    if (n === 0) return;

    const pad = { top: 12, right: 24, bottom: 48, left: 64 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Global min/max across all columns
    let globalMin = Infinity, globalMax = -Infinity;
    for (const s of stats) {
      if (!s) continue;
      const iqr = (s.q3 - s.q1);
      const lo = s.q1 - 1.5 * iqr;
      const hi = s.q3 + 1.5 * iqr;
      globalMin = Math.min(globalMin, s.min, lo);
      globalMax = Math.max(globalMax, s.max, hi);
    }
    if (globalMin === Infinity) return;
    const range = (globalMax - globalMin) || 1;

    const boxW = Math.min(plotW / n * 0.6, 40);
    const gap = plotW / n;

    const COLORS = ['#cfbcff', '#e7c365', '#ff8a80', '#80cbc4', '#b39ddb', '#ffab91', '#81d4fa'];

    const mapY = (v: number) => pad.top + plotH - ((v - globalMin) / range) * plotH;

    for (let i = 0; i < n; i++) {
      const s = stats[i];
      const cx = pad.left + i * gap + gap / 2;
      const color = COLORS[i % COLORS.length];

      const whiskerLo = Math.max(s.min, s.q1 - 1.5 * (s.q3 - s.q1));
      const whiskerHi = Math.min(s.max, s.q3 + 1.5 * (s.q3 - s.q1));

      // Whisker lines
      ctx.strokeStyle = color + '80';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, mapY(whiskerLo));
      ctx.lineTo(cx, mapY(s.q1));
      ctx.moveTo(cx, mapY(s.q3));
      ctx.lineTo(cx, mapY(whiskerHi));
      ctx.stroke();

      // Whisker caps
      ctx.beginPath();
      ctx.moveTo(cx - boxW * 0.3, mapY(whiskerLo));
      ctx.lineTo(cx + boxW * 0.3, mapY(whiskerLo));
      ctx.moveTo(cx - boxW * 0.3, mapY(whiskerHi));
      ctx.lineTo(cx + boxW * 0.3, mapY(whiskerHi));
      ctx.stroke();

      // Box
      const yQ3 = mapY(s.q3);
      const yQ1 = mapY(s.q1);
      ctx.fillStyle = color + '30';
      ctx.fillRect(cx - boxW / 2, yQ3, boxW, yQ1 - yQ3);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.strokeRect(cx - boxW / 2, yQ3, boxW, yQ1 - yQ3);

      // Median line
      const yMed = mapY(s.median);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - boxW / 2, yMed);
      ctx.lineTo(cx + boxW / 2, yMed);
      ctx.stroke();

      // Label
      ctx.font = '400 9px JetBrains Mono';
      ctx.fillStyle = '#948e9c';
      const maxLblLen = gap < 50 ? 5 : gap < 70 ? 7 : 12;
      const lbl = labels[i].length > maxLblLen ? labels[i].slice(0, maxLblLen - 1) + '…' : labels[i];
      if (gap < 50) {
        ctx.save();
        ctx.translate(cx, height - 4);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'right';
        ctx.fillText(lbl, 0, 0);
        ctx.restore();
      } else {
        ctx.textAlign = 'center';
        ctx.fillText(lbl, cx, height - 6);
      }
    }

    // Y axis
    ctx.strokeStyle = 'rgba(148,142,156,0.2)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.stroke();

    const fmt = (n: number) => Math.abs(n) >= 1000 ? (n / 1000).toFixed(1) + 'k' : n.toFixed(1);
    ctx.font = '400 10px JetBrains Mono';
    ctx.fillStyle = '#948e9c';
    ctx.textAlign = 'right';
    ctx.fillText(fmt(globalMin), pad.left - 4, pad.top + plotH);
    ctx.fillText(fmt(globalMax), pad.left - 4, pad.top + 10);
  }, [stats, labels, width, height]);

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />;
}
