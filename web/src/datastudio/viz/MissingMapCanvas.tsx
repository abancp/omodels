/* Canvas-based Missing Values Map */
import { useRef, useEffect } from 'react';

interface Props {
  data: Record<string, unknown>[];
  columns: string[];
  width: number;
  height: number;
}

export default function MissingMapCanvas({ data, columns, width, height }: Props) {
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

    const nCols = columns.length;
    const nRows = data.length;
    if (nCols === 0 || nRows === 0) return;

    const labelPad = 60;
    const topPad = 8;
    const plotW = width - labelPad - 8;
    const plotH = height - topPad - 24;

    const cellW = plotW / nCols;
    // Show max 200 rows, sample otherwise
    const maxRows = Math.min(nRows, 200);
    const cellH = plotH / maxRows;
    const step = Math.max(1, Math.floor(nRows / maxRows));

    for (let ci = 0; ci < nCols; ci++) {
      const col = columns[ci];
      for (let ri = 0; ri < maxRows; ri++) {
        const rowIdx = ri * step;
        if (rowIdx >= nRows) break;
        const val = data[rowIdx][col];
        const isMissing = val == null || val === '' || val === undefined;

        const x = labelPad + ci * cellW;
        const y = topPad + ri * cellH;

        ctx.fillStyle = isMissing ? '#ff8a80' : 'rgba(207,188,255,0.2)';
        ctx.fillRect(x, y, cellW - 1, cellH - 1);
      }
    }

    // Column labels
    ctx.font = '400 8px JetBrains Mono';
    ctx.fillStyle = '#948e9c';
    ctx.textAlign = 'center';
    for (let ci = 0; ci < nCols; ci++) {
      const lbl = columns[ci].length > 7 ? columns[ci].slice(0, 6) + '…' : columns[ci];
      ctx.fillText(lbl, labelPad + ci * cellW + cellW / 2, height - 4);
    }

    // Row count label
    ctx.textAlign = 'right';
    ctx.fillText(`${nRows} rows`, labelPad - 4, topPad + 10);
  }, [data, columns, width, height]);

  return <canvas ref={ref} style={{ width, height, display: 'block' }} />;
}
