/**
 * PipelineCanvas — interactive DAG canvas with pan/zoom and drag-drop
 */

import { useRef, useState, useCallback, useEffect } from 'react';
import { usePipeline, newNodeId, newEdgeId } from '../PipelineStore';
import type { PipelineNode, PipelineEdge, NodeFamily } from '../PipelineTypes';
import { getCatalogEntry, NODE_CATALOG } from '../NodeCatalog';
import ContextMenu from './ContextMenu';
import type { ContextMenuEntry } from './ContextMenu';

const FAMILY_COLORS: Record<string, string> = {
  dataset: '#6366f1',
  split: '#818cf8',
  preprocessor: '#10b981',
  model: '#f59e0b',
  evaluation: '#ef4444',
  output: '#06b6d4',
};

const STATE_COLORS: Record<string, string> = {
  idle: 'var(--c-panel-border)',
  training: '#f59e0b',
  trained: '#10b981',
  stale: '#f59e0b',
  error: '#ef4444',
};

interface ConnectState {
  fromNodeId: string;
  fromX: number;
  fromY: number;
  curX: number;
  curY: number;
}

export default function PipelineCanvas() {
  const {
    pipeline, selectedNodeId, addNode, removeNode, moveNode, selectNode,
    addEdge, removeEdge, isRunning, nodeResults, canConnect,
  } = usePipeline();

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Hover node preview state
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [hoveredNodeRect, setHoveredNodeRect] = useState<DOMRect | null>(null);

  // Pan/zoom state
  const [pan, setPan] = useState({ x: 60, y: 60 });
  const [zoom, setZoom] = useState(1);
  const [isPanningActive, setIsPanningActive] = useState(false);
  const isPanning = useRef(false);
  const lastPan = useRef({ x: 0, y: 0 });

  // Context menu state
  type CtxTarget = { type: 'node'; nodeId: string } | { type: 'canvas'; wx: number; wy: number } | null;
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; target: CtxTarget } | null>(null);
  const closeCtx = useCallback(() => setCtxMenu(null), []);

  // Drag node state
  const draggingNode = useRef<{ id: string; ox: number; oy: number; startX: number; startY: number } | null>(null);
  const hasDragged = useRef(false);
  const dragStartPos = useRef({ x: 0, y: 0 });

  // Connection drawing
  const [connecting, setConnecting] = useState<ConnectState | null>(null);

  // Hover timeout
  const hoverTimeout = useRef<NodeJS.Timeout | null>(null);

  // ─── Coordinate helpers ────────────────────────────────────────────────────

  const screenToWorld = useCallback((sx: number, sy: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return { x: (sx - rect.left - pan.x) / zoom, y: (sy - rect.top - pan.y) / zoom };
  }, [pan, zoom]);

  // ─── Pan ──────────────────────────────────────────────────────────────────

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 2) return; // ignore right-click (for context menus)
    isPanning.current = true;
    setIsPanningActive(true);
    lastPan.current = { x: e.clientX - pan.x, y: e.clientY - pan.y };
    
    // Track click drag detection
    hasDragged.current = false;
    dragStartPos.current = { x: e.clientX, y: e.clientY };
  }, [pan]);

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning.current) {
      const dist = Math.hypot(e.clientX - dragStartPos.current.x, e.clientY - dragStartPos.current.y);
      if (dist > 3) {
        hasDragged.current = true;
      }
      setPan({ x: e.clientX - lastPan.current.x, y: e.clientY - lastPan.current.y });
      return;
    }
    if (draggingNode.current) {
      const dist = Math.hypot(e.clientX - draggingNode.current.startX, e.clientY - draggingNode.current.startY);
      if (dist > 3) {
        hasDragged.current = true;
      }
      const world = screenToWorld(e.clientX, e.clientY);
      moveNode(draggingNode.current.id, {
        x: world.x - draggingNode.current.ox,
        y: world.y - draggingNode.current.oy,
      });
      return;
    }
    if (connecting) {
      setConnecting(c => c ? { ...c, curX: e.clientX, curY: e.clientY } : null);
    }
  }, [connecting, screenToWorld, moveNode]);

  const onMouseUp = useCallback((_e: React.MouseEvent) => {
    isPanning.current = false;
    setIsPanningActive(false);
    draggingNode.current = null;
    if (connecting) setConnecting(null);
  }, [connecting]);

  // ─── Zoom ─────────────────────────────────────────────────────────────────

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom(z => Math.max(0.3, Math.min(2, z * factor)));
  }, []);

  // ─── Drop from palette ────────────────────────────────────────────────────

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const data = e.dataTransfer.getData('application/pipeline-node');
    if (!data) return;
    const { family, type, label } = JSON.parse(data) as { family: NodeFamily; type: string; label: string };
    const catalog = getCatalogEntry(family, type);
    if (!catalog) return;

    const world = screenToWorld(e.clientX, e.clientY);


    const node: PipelineNode = {
      id: newNodeId(),
      family,
      type,
      label,
      config: { ...catalog.defaultConfig } as any,
      state: 'idle',
      position: { x: world.x - 90, y: world.y - 30 },
    };
    addNode(node);
    selectNode(node.id);
  }, [pipeline.nodes, screenToWorld, addNode, selectNode]);

  const onDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); }, []);

  // ─── Node mouse events ────────────────────────────────────────────────────

  const onNodeMouseDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    if (e.button === 2) return; // let onContextMenu handle right-click
    e.stopPropagation();
    selectNode(nodeId);
    
    hasDragged.current = false;
    
    const node = pipeline.nodes.find(n => n.id === nodeId);
    if (!node) return;
    const world = screenToWorld(e.clientX, e.clientY);
    draggingNode.current = {
      id: nodeId,
      ox: world.x - node.position.x,
      oy: world.y - node.position.y,
      startX: e.clientX,
      startY: e.clientY,
    };
  }, [pipeline.nodes, screenToWorld, selectNode]);

  // ─── Context Menus ────────────────────────────────────────────────────────

  const onNodeContextMenu = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.preventDefault();
    e.stopPropagation();
    if (!connecting) {
      selectNode(nodeId);
    }
    setCtxMenu({ x: e.clientX, y: e.clientY, target: { type: 'node', nodeId } });
  }, [selectNode, connecting]);

  const onNodeMouseUp = useCallback((e: React.MouseEvent, nodeId: string) => {
    if (connecting && connecting.fromNodeId !== nodeId) {
      e.stopPropagation();
      if (canConnect(connecting.fromNodeId, nodeId)) {
        addEdge({ id: newEdgeId(), source: connecting.fromNodeId, target: nodeId });
      }
      setConnecting(null);
    }
  }, [connecting, addEdge, canConnect]);

  const onCanvasContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const world = screenToWorld(e.clientX, e.clientY);
    setCtxMenu({ x: e.clientX, y: e.clientY, target: { type: 'canvas', wx: world.x, wy: world.y } });
  }, [screenToWorld]);

  const buildNodeMenu = useCallback((nodeId: string): ContextMenuEntry[] => {
    const node = pipeline.nodes.find(n => n.id === nodeId);
    if (!node) return [];
    const quickAdd = (family: NodeFamily, type: string) => {
      const catalog = getCatalogEntry(family, type);
      if (!catalog) return;
      const newNode: PipelineNode = {
        id: newNodeId(), family, type, label: catalog.label,
        config: { ...catalog.defaultConfig } as any, state: 'idle',
        position: { x: node.position.x + 220, y: node.position.y },
      };
      addNode(newNode);
      addEdge({ id: newEdgeId(), source: nodeId, target: newNode.id });
      selectNode(newNode.id);
    };

    return [
      { id: 'hdr', label: node.label, icon: 'circle', disabled: true },
      { id: 's0', separator: true },
      { id: 'preview', label: 'Preview Data', icon: 'visibility',
        disabled: node.state !== 'trained',
        action: () => { selectNode(nodeId); window.dispatchEvent(new CustomEvent('pipeline:preview-tab', { detail: nodeId })); } },
      { id: 'inspect', label: 'Inspect Config', icon: 'settings',
        action: () => { selectNode(nodeId); window.dispatchEvent(new CustomEvent('pipeline:config-tab', { detail: nodeId })); } },
      { id: 's1', separator: true },
      { id: 'duplicate', label: 'Duplicate', icon: 'content_copy',
        action: () => {
          const dup: PipelineNode = {
            ...node, id: newNodeId(),
            position: { x: node.position.x + 30, y: node.position.y + 40 },
            state: 'idle', fittedParams: undefined, metrics: undefined, error: undefined,
          };
          addNode(dup);
          selectNode(dup.id);
        } },
      { id: 'chain', label: 'Add connected node', icon: 'add_circle',
        children: NODE_CATALOG
          .filter(e => e.family !== 'dataset')
          .slice(0, 12)
          .map(e => ({
            id: `chain-${e.type}`, label: e.label, icon: e.icon,
            action: () => quickAdd(e.family, e.type),
          })) },
      { id: 's2', separator: true },
      { id: 'disconnect', label: 'Disconnect all edges', icon: 'link_off',
        action: () => {
          pipeline.edges.filter(e => e.source === nodeId || e.target === nodeId)
            .forEach(e => removeEdge(e.id));
        } },
      { id: 'delete', label: 'Delete node', icon: 'delete', danger: true,
        action: () => removeNode(nodeId) },
    ];
  }, [pipeline, addNode, addEdge, removeEdge, removeNode, selectNode]);

  const buildCanvasMenu = useCallback((wx: number, wy: number): ContextMenuEntry[] => {
    const spawnNode = (family: NodeFamily, type: string) => {
      const catalog = getCatalogEntry(family, type);
      if (!catalog) return;
      const node: PipelineNode = {
        id: newNodeId(), family, type, label: catalog.label,
        config: { ...catalog.defaultConfig } as any, state: 'idle',
        position: { x: wx - 90, y: wy - 32 },
      };
      addNode(node); selectNode(node.id);
    };

    return [
      { id: 'add', label: 'Add Node', icon: 'add_circle',
        children: [
          { id: 'add-ds', label: 'Dataset', icon: 'dataset', action: () => spawnNode('dataset', 'Dataset') },
          { id: 'add-split', label: 'Train/Test Split', icon: 'call_split', action: () => spawnNode('split', 'TrainTestSplit') },
          { id: 'add-scaler', label: 'Standard Scaler', icon: 'equalizer', action: () => spawnNode('preprocessor', 'StandardScaler') },
          { id: 'add-label', label: 'Label Encoder', icon: 'font_download', action: () => spawnNode('preprocessor', 'LabelEncoder') },
          { id: 'add-onehot', label: 'One-Hot Encoder', icon: 'grid_on', action: () => spawnNode('preprocessor', 'OneHotEncoder') },
          { id: 'add-lr', label: 'Linear Regression', icon: 'trending_up', action: () => spawnNode('model', 'LinearRegression') },
          { id: 'add-log', label: 'Logistic Regression', icon: 'scatter_plot', action: () => spawnNode('model', 'LogisticRegression') },
          { id: 'add-knn', label: 'KNN', icon: 'grain', action: () => spawnNode('model', 'KNN') },
          { id: 'add-eval-cls', label: 'Classification Metrics', icon: 'assessment', action: () => spawnNode('evaluation', 'ClassificationMetrics') },
          { id: 'add-eval-reg', label: 'Regression Metrics', icon: 'analytics', action: () => spawnNode('evaluation', 'RegressionMetrics') },
        ] },
      { id: 's0', separator: true },
      { id: 'fit', label: 'Fit View', icon: 'fit_screen',
        action: () => { setPan({ x: 60, y: 60 }); setZoom(1); } },
      { id: 'zoom-in', label: 'Zoom In', icon: 'zoom_in', shortcut: 'Scroll ↑',
        action: () => setZoom(z => Math.min(2, z * 1.2)) },
      { id: 'zoom-out', label: 'Zoom Out', icon: 'zoom_out', shortcut: 'Scroll ↓',
        action: () => setZoom(z => Math.max(0.3, z * 0.83)) },
      { id: 's1', separator: true },
      { id: 'clear', label: 'Clear All Nodes', icon: 'clear_all', danger: true,
        disabled: pipeline.nodes.length === 0,
        action: () => { if (confirm('Clear all nodes?')) {
          pipeline.nodes.forEach(n => removeNode(n.id));
        }}},
    ];
  }, [pipeline, addNode, removeNode, selectNode, setPan, setZoom]);

  // ─── Keyboard Shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input
      if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA' || document.activeElement?.tagName === 'SELECT') {
        return;
      }

      // Delete Node
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNodeId) {
        e.preventDefault();
        removeNode(selectedNodeId);
        selectNode(null);
      }
      
      // Copy Node (Ctrl+C)
      if ((e.ctrlKey || e.metaKey) && e.key === 'c' && selectedNodeId) {
        e.preventDefault();
        const node = pipeline.nodes.find(n => n.id === selectedNodeId);
        if (node) {
          localStorage.setItem('om_clipboard', JSON.stringify(node));
        }
      }
      
      // Paste Node (Ctrl+V)
      if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        try {
          const data = localStorage.getItem('om_clipboard');
          if (data) {
            const parsed = JSON.parse(data) as PipelineNode;
            if (parsed.family === 'dataset' && pipeline.nodes.some(n => n.family === 'dataset')) return;
            const nn: PipelineNode = {
              ...parsed,
              id: newNodeId(),
              position: { x: parsed.position.x + 30, y: parsed.position.y + 30 },
            };
            addNode(nn);
            selectNode(nn.id);
          }
        } catch(err) { /* ignore */ }
      }
      
      // Duplicate Node (Ctrl+D)
      if ((e.ctrlKey || e.metaKey) && e.key === 'd' && selectedNodeId) {
        e.preventDefault();
        const node = pipeline.nodes.find(n => n.id === selectedNodeId);
        if (node) {
          if (node.family === 'dataset') return; // Multiple datasets allowed, skip duplicate for now
          const nn: PipelineNode = {
            ...node,
            id: newNodeId(),
            position: { x: node.position.x + 30, y: node.position.y + 30 },
          };
          addNode(nn);
          selectNode(nn.id);
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedNodeId, pipeline, removeNode, selectNode, addNode]);

  // ─── Port events (connection handles) ────────────────────────────────────

  const onOutputPortDown = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    setConnecting({
      fromNodeId: nodeId,
      fromX: e.clientX,
      fromY: e.clientY,
      curX: e.clientX,
      curY: e.clientY,
    });
  }, []);

  const onInputPortUp = useCallback((e: React.MouseEvent, nodeId: string) => {
    e.stopPropagation();
    if (!connecting) return;
    if (connecting.fromNodeId === nodeId) { setConnecting(null); return; }

    if (canConnect(connecting.fromNodeId, nodeId)) {
      const edge: PipelineEdge = {
        id: newEdgeId(),
        source: connecting.fromNodeId,
        target: nodeId,
      };
      addEdge(edge);
    }
    setConnecting(null);
  }, [connecting, addEdge, canConnect]);

  // ─── Node geometry ─────────────────────────────────────────────────────────

  const NODE_W = 180;
  const NODE_H = 64;


  function outputPortPos(node: PipelineNode) {
    return { x: node.position.x + NODE_W, y: node.position.y + NODE_H / 2 };
  }

  function inputPortPos(node: PipelineNode) {
    return { x: node.position.x, y: node.position.y + NODE_H / 2 };
  }

  // ─── Edge path ─────────────────────────────────────────────────────────────

  function edgePath(x1: number, y1: number, x2: number, y2: number): string {
    const cx = (x1 + x2) / 2;
    return `M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}`;
  }

  // ─── Render ────────────────────────────────────────────────────────────────

  const nodeMap = new Map(pipeline.nodes.map(n => [n.id, n]));

  const transform = `translate(${pan.x},${pan.y}) scale(${zoom})`;

  return (
    <div
      ref={containerRef}
      className="pipeline-canvas"
      style={{ cursor: isPanningActive ? 'grabbing' : 'grab', userSelect: 'none', WebkitUserSelect: 'none' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onWheel={onWheel}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onClick={() => { if (!hasDragged.current) { selectNode(null); closeCtx(); } }}
      onContextMenu={onCanvasContextMenu}
    >
      {/* Grid background */}
      <svg
        ref={svgRef}
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', overflow: 'visible' }}
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern id="grid" width={20 * zoom} height={20 * zoom} patternUnits="userSpaceOnUse"
            x={pan.x % (20 * zoom)} y={pan.y % (20 * zoom)}>
            <circle cx={1} cy={1} r={0.8} fill="var(--c-panel-border)" opacity="0.5" />
          </pattern>
          <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="var(--c-on-surface-variant)" opacity="0.6" />
          </marker>
          <marker id="arrowhead-active" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="var(--c-primary)" />
          </marker>
        </defs>

        <rect width="100%" height="100%" fill="url(#grid)" />

        <g transform={transform}>
          {/* Edges */}
          {pipeline.edges.map(edge => {
            const src = nodeMap.get(edge.source);
            const tgt = nodeMap.get(edge.target);
            if (!src || !tgt) return null;
            const op = outputPortPos(src);
            const ip = inputPortPos(tgt);
            const isSelected = selectedNodeId === edge.source || selectedNodeId === edge.target;
            return (
              <g key={edge.id}>
                <title>Click to delete edge</title>
                <path
                  d={edgePath(op.x, op.y, ip.x, ip.y)}
                  fill="none"
                  stroke={isSelected ? 'var(--c-primary)' : 'var(--c-on-surface-variant)'}
                  strokeWidth={isSelected ? 2 / zoom : 1.5 / zoom}
                  strokeOpacity={isSelected ? 1 : 0.4}
                  markerEnd={isSelected ? 'url(#arrowhead-active)' : 'url(#arrowhead)'}
                  style={{ cursor: 'pointer', transition: 'stroke 0.15s' }}
                  onClick={e => { e.stopPropagation(); removeEdge(edge.id); }}
                />
              </g>
            );
          })}

          {/* Live connection line */}
          {connecting && (() => {
            const src = nodeMap.get(connecting.fromNodeId);
            if (!src) return null;
            const op = outputPortPos(src);
            const rect = containerRef.current?.getBoundingClientRect();
            if (!rect) return null;
            const wx = (connecting.curX - rect.left - pan.x) / zoom;
            const wy = (connecting.curY - rect.top - pan.y) / zoom;
            return (
              <path
                d={edgePath(op.x, op.y, wx, wy)}
                fill="none"
                stroke="var(--c-primary)"
                strokeWidth={2 / zoom}
                strokeDasharray={`${6 / zoom},${4 / zoom}`}
                markerEnd="url(#arrowhead-active)"
              />
            );
          })()}

          {/* Nodes */}
          {pipeline.nodes.map(node => {
            const color = FAMILY_COLORS[node.family] ?? '#888';
            const stateColor = STATE_COLORS[node.state] ?? 'transparent';
            const isSelected = selectedNodeId === node.id;
            const isSource = node.family === 'dataset';
            const isOutput = node.family === 'output' || node.family === 'evaluation';

            return (
              <g
                key={node.id}
                transform={`translate(${node.position.x},${node.position.y})`}
                style={{ cursor: connecting ? 'crosshair' : 'grab' }}
                onMouseDown={e => onNodeMouseDown(e, node.id)}
                onMouseUp={e => onNodeMouseUp(e, node.id)}
                onClick={e => e.stopPropagation()}
                onContextMenu={e => onNodeContextMenu(e, node.id)}
                onMouseEnter={e => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
                  hoverTimeout.current = setTimeout(() => {
                    setHoveredNodeId(node.id);
                    setHoveredNodeRect(rect);
                  }, 400);
                }}
                onMouseLeave={() => {
                  if (hoverTimeout.current) clearTimeout(hoverTimeout.current);
                  setHoveredNodeId(null);
                  setHoveredNodeRect(null);
                }}
              >
                {/* Shadow */}
                <rect
                  x={2} y={3} width={NODE_W} height={NODE_H}
                  rx={8} fill="rgba(0,0,0,0.4)"
                />
                {/* Body */}
                <rect
                  x={0} y={0} width={NODE_W} height={NODE_H}
                  rx={8}
                  fill={isSelected
                    ? `color-mix(in srgb, ${color} 15%, var(--c-surface-container))`
                    : 'var(--c-surface-container)'}
                  stroke={isSelected ? color : 'var(--c-panel-border)'}
                  strokeWidth={isSelected ? 2 : 1}
                />
                {/* Left color bar */}
                <rect x={0} y={0} width={4} height={NODE_H} rx={8} fill={color} />
                <rect x={0} y={8} width={4} height={NODE_H - 16} fill={color} />

                {/* State indicator dot */}
                {node.state !== 'idle' && (
                  <circle cx={NODE_W - 12} cy={12} r={5} fill={stateColor}>
                    {node.state === 'training' && (
                      <animate attributeName="opacity" values="1;0.3;1" dur="1s" repeatCount="indefinite" />
                    )}
                  </circle>
                )}

                {/* Label */}
                <text
                  x={16} y={26}
                  fill="var(--c-on-surface)"
                  fontSize={11 / zoom > 13 ? 13 : 11}
                  fontFamily="Inter, sans-serif"
                  fontWeight={600}
                >
                  {node.label.length > 20 ? node.label.slice(0, 18) + '…' : node.label}
                </text>

                {/* Type badge */}
                <text
                  x={16} y={42}
                  fill={color}
                  fontSize={9}
                  fontFamily="JetBrains Mono, monospace"
                  opacity={0.85}
                >
                  {node.family.toUpperCase()}
                </text>

                {/* Metrics quick view */}
                {node.metrics && Object.keys(node.metrics).length > 0 && (
                  <text x={16} y={56} fill="var(--c-on-surface-variant)" fontSize={9} fontFamily="JetBrains Mono, monospace">
                    {Object.entries(node.metrics).slice(0, 2).map(([k, v]) => `${k}: ${(v as number).toFixed(3)}`).join('  ')}
                  </text>
                )}

                {/* Error badge */}
                {node.state === 'error' && node.error && (
                  <text x={16} y={56} fill="#ef4444" fontSize={9} fontFamily="Inter, sans-serif">
                    ✗ {node.error.slice(0, 28)}
                  </text>
                )}

                {/* Input port */}
                {!isSource && (
                  <circle
                    cx={0} cy={NODE_H / 2} r={6}
                    fill="var(--c-surface-container-highest)"
                    stroke={color}
                    strokeWidth={2}
                    style={{ cursor: 'crosshair' }}
                    onMouseUp={e => onInputPortUp(e, node.id)}
                  />
                )}

                {/* Output port */}
                {!isOutput && (
                  <circle
                    cx={NODE_W} cy={NODE_H / 2} r={6}
                    fill="var(--c-surface-container-highest)"
                    stroke={color}
                    strokeWidth={2}
                    style={{ cursor: 'crosshair' }}
                    onMouseDown={e => { e.stopPropagation(); onOutputPortDown(e, node.id); }}
                  />
                )}

                {/* Delete button on hover */}
                {isSelected && (
                  <g
                    transform={`translate(${NODE_W - 22},${-10})`}
                    style={{ cursor: 'pointer' }}
                    onClick={e => { e.stopPropagation(); removeNode(node.id); }}
                  >
                    <circle cx={10} cy={10} r={10} fill="#ef4444" />
                    <text x={5} y={14} fill="white" fontSize={12} fontWeight={700}>✕</text>
                  </g>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {/* Empty state */}
      {pipeline.nodes.length === 0 && (
        <div className="pipeline-canvas__empty">
          <div className="pipeline-canvas__empty-icon">⬡</div>
          <div className="pipeline-canvas__empty-title">Start building your pipeline</div>
          <div className="pipeline-canvas__empty-sub">Drag nodes from the left panel onto the canvas</div>
        </div>
      )}

      {/* Zoom controls */}
      <div className="pipeline-canvas__controls">
        <button className="pipeline-canvas__ctrl-btn" onClick={() => setZoom(z => Math.min(2, z * 1.2))} title="Zoom in">+</button>
        <button className="pipeline-canvas__ctrl-btn pipeline-canvas__ctrl-btn--percent" onClick={() => setZoom(1)} title="Reset zoom">{Math.round(zoom * 100)}%</button>
        <button className="pipeline-canvas__ctrl-btn" onClick={() => setZoom(z => Math.max(0.3, z / 1.2))} title="Zoom out">−</button>
        <button className="pipeline-canvas__ctrl-btn" onClick={() => { setPan({ x: 60, y: 60 }); setZoom(1); }} title="Fit">⊡</button>
      </div>

      {/* Status bar */}
      <div className="pipeline-canvas__status">
        <div className={`pipeline-canvas__state pipeline-canvas__state--${pipeline.state}`}>
          {pipeline.state.toUpperCase()}
        </div>
        <span className="pipeline-canvas__status-info">
          {pipeline.nodes.length} nodes · {pipeline.edges.length} edges
        </span>
        {isRunning && <span className="pipeline-canvas__running">● Running…</span>}
      </div>

      {/* Context Menu */}
      {ctxMenu && (
        <ContextMenu
          x={ctxMenu.x}
          y={ctxMenu.y}
          items={
            ctxMenu.target?.type === 'node'
              ? buildNodeMenu(ctxMenu.target.nodeId)
              : buildCanvasMenu((ctxMenu.target as any)?.wx ?? 200, (ctxMenu.target as any)?.wy ?? 200)
          }
          onClose={closeCtx}
        />
      )}

      {/* Dynamic Hover Preview Card */}
      {hoveredNodeId && hoveredNodeRect && (() => {
        const hNode = pipeline.nodes.find(n => n.id === hoveredNodeId);
        if (!hNode) return null;
        const color = FAMILY_COLORS[hNode.family] ?? '#888';
        const result = nodeResults[hNode.id];
        const hasMetrics = result?.metrics && Object.keys(result.metrics).length > 0;

        return (
          <div
            style={{
              position: 'fixed',
              left: hoveredNodeRect.right + 12,
              top: Math.max(10, Math.min(window.innerHeight - 360, hoveredNodeRect.top - 20)),
              width: 290,
              maxHeight: 480,
              background: 'var(--c-surface-container-highest, #1e1e2e)',
              border: '1.5px solid',
              borderColor: color,
              borderRadius: 8,
              padding: 12,
              boxShadow: '0 12px 40px rgba(0,0,0,0.6), 0 2px 10px rgba(0,0,0,0.3)',
              zIndex: 10000,
              overflowY: 'auto',
              fontFamily: 'Inter, sans-serif',
              pointerEvents: 'none',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{
                fontSize: 8,
                textTransform: 'uppercase',
                fontWeight: 750,
                background: 'rgba(255,255,255,0.06)',
                padding: '1px 5px',
                borderRadius: 3,
                color,
              }}>
                {hNode.family}
              </span>
              <strong style={{ fontSize: 11, color: 'var(--c-on-surface)', fontWeight: 600 }}>
                {hNode.label}
              </strong>
              <span
                style={{
                  marginLeft: 'auto',
                  fontSize: 9,
                  fontWeight: 600,
                  color: hNode.state === 'trained' ? '#10b981' : hNode.state === 'error' ? '#ef4444' : '#f59e0b',
                }}
              >
                ● {hNode.state.toUpperCase()}
              </span>
            </div>

            {/* DATASET DETAILS */}
            {hNode.family === 'dataset' && (() => {
              const cfg = hNode.config as any;
              return (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 10, color: 'var(--c-on-surface-variant)' }}>
                  {cfg.csvFilename ? (
                    <>
                      <div>📁 File: <strong>{cfg.csvFilename}</strong></div>
                      <div>📊 Size: <strong>{cfg.csvRowCount ?? 0} rows · {cfg.csvHeaders?.length ?? 0} columns</strong></div>
                      <div>🎯 Target: <strong style={{ color: 'var(--c-primary)' }}>{cfg.targetColumn || 'Not selected'}</strong></div>
                      <div>⚙ Task: <strong>{cfg.taskType || 'Auto-detect'}</strong></div>
                    </>
                  ) : (
                    <div style={{ opacity: 0.5, fontStyle: 'italic' }}>No CSV uploaded. Select this node and upload data.</div>
                  )}
                </div>
              );
            })()}

            {/* PREPROCESSOR DETAILS */}
            {hNode.family === 'preprocessor' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {result ? (
                  <>
                    <div style={{ fontSize: 9, fontFamily: 'monospace', color: 'var(--c-on-surface-variant)' }}>
                      In: <span style={{ color: 'var(--c-primary)' }}>[{result.inputSnapshot?.shape.join(' × ')}]</span> · Out: <span style={{ color: '#10b981' }}>[{result.outputSnapshot?.shape.join(' × ')}]</span>
                    </div>
                    {/* Fitted parameters overview */}
                    {result.fittedParams && (() => {
                      const params = result.fittedParams;
                      if (hNode.type === 'StandardScaler' && params.means) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Fitted μ & σ (first 2 features):</div>
                            {params.means.slice(0, 2).map((m: number, idx: number) => (
                              <div key={idx} style={{ opacity: 0.8 }}>
                                • Feat {idx}: mean={m.toFixed(2)}, std={params.stds[idx]?.toFixed(2)}
                              </div>
                            ))}
                          </div>
                        );
                      }
                      if (hNode.type === 'MinMaxScaler' && params.mins) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Bounds:</div>
                            {params.mins.slice(0, 2).map((m: number, idx: number) => (
                              <div key={idx} style={{ opacity: 0.8 }}>
                                • Feat {idx}: [{m.toFixed(2)}, {params.maxs[idx]?.toFixed(2)}]
                              </div>
                            ))}
                          </div>
                        );
                      }
                      if (hNode.type === 'DropColumns' && params.droppedColumns) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Dropped Columns:</div>
                            <div style={{ opacity: 0.8 }}>
                              {params.droppedColumns.length > 0 ? params.droppedColumns.join(', ') : 'None'}
                            </div>
                          </div>
                        );
                      }
                      if (hNode.type === 'SimpleImputer' && params.values) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Imputed values (Strategy: {params.strategy}):</div>
                            {params.values.slice(0, 2).map((v: number, idx: number) => (
                              <div key={idx} style={{ opacity: 0.8 }}>
                                • Column {idx}: {v.toFixed(3)}
                              </div>
                            ))}
                          </div>
                        );
                      }
                      if (hNode.type === 'PCA' && params.explainedVariance) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Eigenvalues (components):</div>
                            {params.explainedVariance.slice(0, 2).map((v: number, idx: number) => (
                              <div key={idx} style={{ opacity: 0.8 }}>
                                • PC{idx + 1}: {v.toFixed(3)}
                              </div>
                            ))}
                          </div>
                        );
                      }
                      return null;
                    })()}
                  </>
                ) : (
                  <div style={{ fontSize: 9, opacity: 0.6, fontStyle: 'italic' }}>
                    Preprocessor idle. Waiting for pipeline run.
                  </div>
                )}
              </div>
            )}

            {/* MODEL DETAILS */}
            {hNode.family === 'model' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {result ? (
                  <>
                    <div style={{ fontSize: 9, fontFamily: 'monospace', color: 'var(--c-on-surface-variant)' }}>
                      Fitted on features: <span style={{ color: 'var(--c-primary)' }}>{result.inputSnapshot?.featureNames.slice(0, 3).join(', ')}{(result.inputSnapshot?.featureNames?.length ?? 0) > 3 ? '…' : ''}</span>
                    </div>
                    {/* Metrics quick glance */}
                    {hasMetrics && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                        {Object.entries(result.metrics ?? {}).slice(0, 3).map(([k, v]) => (
                          <span
                            key={k}
                            style={{
                              fontSize: 8,
                              fontFamily: 'monospace',
                              background: 'rgba(245, 158, 11, 0.1)',
                              border: '0.5px solid rgba(245, 158, 11, 0.3)',
                              color: '#f59e0b',
                              padding: '2px 5px',
                              borderRadius: 4,
                            }}
                          >
                            {k}: {typeof v === 'number' ? v.toFixed(3) : String(v)}
                          </span>
                        ))}
                      </div>
                    )}
                    {/* Model-specific weights/depth parameters */}
                    {result.fittedParams && (() => {
                      const params = result.fittedParams;
                      if (params.weights) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Weights/Coefficients (first 2):</div>
                            {params.weights.slice(0, 2).map((w: number, idx: number) => (
                              <div key={idx} style={{ opacity: 0.8 }}>
                                • Feat {idx}: {w >= 0 ? '+' : ''}{w.toFixed(4)}
                              </div>
                            ))}
                            {params.bias !== undefined && <div style={{ opacity: 0.8 }}>• Bias (Intercept): {params.bias.toFixed(4)}</div>}
                          </div>
                        );
                      }
                      if (params.root) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Split Tree structure:</div>
                            <div style={{ opacity: 0.8 }}>Root threshold split detected. Check the Node Preview tab for split paths!</div>
                          </div>
                        );
                      }
                      if (params.hiddenLayers) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>Neural Network:</div>
                            <div style={{ opacity: 0.8 }}>• Architecture: {params.hiddenLayers.join(' → ')}</div>
                            {params.lossHistory && params.lossHistory.length > 0 && (
                              <div style={{ opacity: 0.8 }}>• Final Loss: {params.lossHistory[params.lossHistory.length - 1].toFixed(5)}</div>
                            )}
                          </div>
                        );
                      }
                      if (params.centroids) {
                        return (
                          <div style={{ fontSize: 9, background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 4 }}>
                            <div style={{ fontWeight: 600, marginBottom: 2 }}>K-Means Cluster Centroids:</div>
                            <div style={{ opacity: 0.8 }}>• K count: {params.nClusters}</div>
                            <div style={{ opacity: 0.8 }}>• Centers: [{params.centroids[0]?.slice(0, 2).map((v: number) => v.toFixed(2)).join(', ')}…]</div>
                          </div>
                        );
                      }
                      return null;
                    })()}
                  </>
                ) : (
                  <div style={{ fontSize: 9, opacity: 0.6, fontStyle: 'italic' }}>
                    Model parameters not fitted. Click Run Pipeline to train.
                  </div>
                )}
              </div>
            )}

            {/* EVALUATION DETAILS */}
            {hNode.family === 'evaluation' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {result ? (
                  <>
                    <div style={{ fontSize: 9, fontFamily: 'monospace', color: 'var(--c-on-surface-variant)' }}>
                      Evaluated on test set: <span style={{ color: 'var(--c-primary)' }}>{result.inputSnapshot?.shape[0] ?? 0} samples</span>
                    </div>
                    {hasMetrics && (
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
                        {Object.entries(result.metrics ?? {}).map(([k, v]) => {
                          if (['tp', 'fp', 'fn', 'tn'].includes(k)) return null;
                          return (
                            <div key={k} style={{ fontSize: 9, display: 'flex', justifyContent: 'space-between', borderBottom: '0.5px solid rgba(255,255,255,0.04)', paddingBottom: 2 }}>
                              <span style={{ opacity: 0.7 }}>{k}:</span>
                              <strong style={{ color: '#ef4444' }}>{typeof v === 'number' ? v.toFixed(4) : String(v)}</strong>
                            </div>
                          );
                        })}
                      </div>
                    )}
                    {/* Confusion matrix values */}
                    {result.metrics?.tp !== undefined && (
                      <div style={{ fontSize: 8, fontFamily: 'monospace', display: 'flex', gap: 4, justifyContent: 'space-around', background: 'rgba(0,0,0,0.15)', padding: 3, borderRadius: 4 }}>
                        <span>TN: {result.metrics.tn}</span>
                        <span>FP: {result.metrics.fp}</span>
                        <span>FN: {result.metrics.fn}</span>
                        <span>TP: {result.metrics.tp}</span>
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ fontSize: 9, opacity: 0.6, fontStyle: 'italic' }}>
                    Evaluation idle. Click Run Pipeline to compute metrics.
                  </div>
                )}
              </div>
            )}

            {/* ERROR VIEW */}
            {hNode.state === 'error' && hNode.error && (
              <div style={{ fontSize: 9, color: '#ef4444', background: 'rgba(239, 68, 68, 0.08)', border: '0.5px solid rgba(239, 68, 68, 0.2)', padding: 5, borderRadius: 4, fontFamily: 'monospace' }}>
                Error: {hNode.error}
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}
