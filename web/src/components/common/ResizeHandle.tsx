/**
 * ResizeHandle — thin draggable strip placed between panels.
 * Emits delta pixels on drag. Supports double-click to reset.
 */

import { useRef, useCallback, type MouseEvent as RMouseEvent } from 'react';

interface ResizeHandleProps {
  /** 'vertical' means a vertical line between left/right panels (drags horizontally) */
  direction: 'vertical';
  onResize: (delta: number) => void;
  onDoubleClick?: () => void;
}

export default function ResizeHandle({ direction, onResize, onDoubleClick }: ResizeHandleProps) {
  const startRef = useRef(0);

  const handleMouseDown = useCallback((e: RMouseEvent) => {
    e.preventDefault();
    startRef.current = direction === 'vertical' ? e.clientX : e.clientY;

    const handleMove = (ev: globalThis.MouseEvent) => {
      const current = direction === 'vertical' ? ev.clientX : ev.clientY;
      const delta = current - startRef.current;
      if (delta !== 0) {
        onResize(delta);
        startRef.current = current;
      }
    };

    const handleUp = () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = direction === 'vertical' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
  }, [direction, onResize]);

  return (
    <div
      className={`resize-handle resize-handle--${direction}`}
      onMouseDown={handleMouseDown}
      onDoubleClick={onDoubleClick}
    />
  );
}
