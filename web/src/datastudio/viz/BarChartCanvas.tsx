/* Canvas-based Bar Chart for categorical data */
import { useRef, useEffect } from 'react';

interface Props {
  labels: string[];
  values: number[];
  width: number;
  height: number;
  accentColor?: string;
  title?: string;
}

export default function BarChartCanvas({ labels, values, width, height, accentColor = '#cfbcff', title }: Props) {
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

    const n = labels.length;
    if (n === 0) return;

    const pad = { top: title ? 24 : 12, right: 24, bottom: 48, left: 64 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const maxVal = Math.max(...values);

    if (title) {
      ctx.font = '600 10px Inter';
      ctx.fillStyle = '#948e9c';
      ctx.textAlign = 'left';
      ctx.fillText(title.toUpperCase(), pad.left, 16);
    }

    const barW = Math.min(plotW / n * 0.7, 30);
    const gap = plotW / n;

    const COLORS = ['#cfbcff', '#e7c365', '#ff8a80', '#80cbc4', '#b39ddb', '#ffab91', '#81d4fa'];

    for (let i = 0; i < n; i++) {
      const barH = maxVal > 0 ? (values[i] / maxVal) * plotH : 0;
      const x = pad.left + i * gap + (gap - barW) / 2;
      const y = pad.top + plotH - barH;
      ctx.fillStyle = COLORS[i % COLORS.length] + 'cc';
      ctx.fillRect(x, y, barW, barH);

      // Value on top
      if (barH > 14) {
        ctx.font = '400 9px JetBrains Mono';
        ctx.fillStyle = '#e6e0e9';
        ctx.textAlign = 'center';
        ctx.fillText(String(values[i]), x + barW / 2, y - 4);
      }

      // Label below — rotate when bars are narrow
      ctx.font = '400 9px JetBrains Mono';
      ctx.fillStyle = '#948e9c';
      const maxLblLen = gap < 40 ? 5 : gap < 60 ? 7 : 12;
      const lbl = labels[i].length > maxLblLen ? labels[i].slice(0, maxLblLen - 1) + '…' : labels[i];
      if (gap < 40) {
        // Rotate labels to avoid overlap
        ctx.save();
        ctx.translate(x + barW / 2, height - 6);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'right';
        ctx.fillText(lbl, 0, 0);
        ctx.restore();
      } else {
        ctx.textAlign = 'center';
        ctx.fillText(lbl, x + barW / 2, height - 8);
      }
    }

    // Y axis
    ctx.strokeStyle = 'rgba(148,142,156,0.2)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    ctx.font = '400 10px JetBrains Mono';
    ctx.fillStyle = '#948e9c';
    ctx.textAlign = 'right';
    ctx.fillText(String(maxVal), pad.left - 4, pad.top + 10);
    ctx.fillText('0', pad.left - 4, pad.top + plotH);
  }, [labels, values, width, height, accentColor, title]);

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />;
}
