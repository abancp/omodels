/**
 * NodePalette — left panel listing all draggable node types
 * Displays a highly polished educational hover tooltip with mathematics, operational flow, and guidelines.
 */

import { useState } from 'react';
import { NODE_CATALOG, FAMILY_META } from '../NodeCatalog';
import type { NodeCatalogEntry } from '../PipelineTypes';
import type { NodeFamily } from '../PipelineTypes';
import Icon from '../../components/common/Icon';
import { THEORY_CATALOG } from './NodeInspector';

const FAMILIES: NodeFamily[] = ['dataset', 'split', 'preprocessor', 'model', 'evaluation', 'output'];

export default function NodePalette() {
  const [search, setSearch] = useState('');
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [hoveredNode, setHoveredNode] = useState<{ entry: NodeCatalogEntry; rect: DOMRect } | null>(null);

  const toggle = (f: string) => setCollapsed(s => {
    const n = new Set(s);
    n.has(f) ? n.delete(f) : n.add(f);
    return n;
  });

  const filtered = search
    ? NODE_CATALOG.filter(n =>
        n.label.toLowerCase().includes(search.toLowerCase()) ||
        n.family.toLowerCase().includes(search.toLowerCase())
      )
    : null;

  return (
    <aside className="pipeline-palette">
      <div className="pipeline-palette__header">
        <div className="pipeline-palette__title">Node Library</div>
        <div className="pipeline-palette__search">
          <Icon name="search" size={12} />
          <input
            type="text"
            placeholder="Search nodes…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pipeline-palette__search-input"
          />
        </div>
      </div>

      <div className="pipeline-palette__body">
        {filtered ? (
          <div className="pipeline-palette__group">
            {filtered.map(entry => (
              <PaletteNode
                key={`${entry.family}-${entry.type}`}
                entry={entry}
                onHoverEnter={(rect) => setHoveredNode({ entry, rect })}
                onHoverLeave={() => setHoveredNode(null)}
              />
            ))}
            {filtered.length === 0 && (
              <div className="pipeline-palette__empty">No nodes found</div>
            )}
          </div>
        ) : (
          FAMILIES.map(family => {
            const meta = FAMILY_META[family];
            const nodes = NODE_CATALOG.filter(n => n.family === family);
            const isOpen = !collapsed.has(family);
            return (
              <div key={family} className="pipeline-palette__section">
                <button
                  className="pipeline-palette__section-header"
                  onClick={() => toggle(family)}
                  style={{ '--accent': meta.color } as any}
                >
                  <span className="pipeline-palette__section-dot" style={{ background: meta.color }} />
                  <span className="pipeline-palette__section-label">{meta.label}</span>
                  <span className="pipeline-palette__section-count">{nodes.length}</span>
                  <span className="pipeline-palette__section-arrow">{isOpen ? '▾' : '▸'}</span>
                </button>
                {isOpen && (
                  <div className="pipeline-palette__group">
                    {nodes.map(entry => (
                      <PaletteNode
                        key={entry.type}
                        entry={entry}
                        onHoverEnter={(rect) => setHoveredNode({ entry, rect })}
                        onHoverLeave={() => setHoveredNode(null)}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      <div className="pipeline-palette__footer">
        <div className="pipeline-palette__tip">
          <Icon name="touch_app" size={12} />
          <span>Drag nodes onto canvas</span>
        </div>
        <div className="pipeline-palette__tip">
          <Icon name="cable" size={12} />
          <span>Drag output → input ports</span>
        </div>
        <div className="pipeline-palette__tip">
          <Icon name="alt_route" size={12} />
          <span>Alt+drag or scroll to pan</span>
        </div>
      </div>

      {/* RENDER THE DETAILED HOVER TOOLTIP */}
      {hoveredNode && (
        <PaletteTooltip entry={hoveredNode.entry} rect={hoveredNode.rect} />
      )}
    </aside>
  );
}

interface PaletteNodeProps {
  entry: NodeCatalogEntry;
  onHoverEnter: (rect: DOMRect) => void;
  onHoverLeave: () => void;
}

function PaletteNode({ entry, onHoverEnter, onHoverLeave }: PaletteNodeProps) {
  const onDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData('application/pipeline-node', JSON.stringify({
      family: entry.family,
      type: entry.type,
      label: entry.label,
    }));
    e.dataTransfer.effectAllowed = 'copy';
  };

  return (
    <div
      className="pipeline-palette__node"
      draggable
      onDragStart={onDragStart}
      onMouseEnter={(e) => onHoverEnter(e.currentTarget.getBoundingClientRect())}
      onMouseLeave={onHoverLeave}
      style={{ '--node-color': entry.color } as any}
    >
      <span className="pipeline-palette__node-icon" style={{ color: entry.color }}>
        <Icon name={entry.icon} size={14} />
      </span>
      <span className="pipeline-palette__node-label">{entry.label}</span>
    </div>
  );
}

function PaletteTooltip({ entry, rect }: { entry: NodeCatalogEntry; rect: DOMRect }) {
  const theory = THEORY_CATALOG[entry.type];

  // Align with hovered list item vertically, and position at the right side of the left panel (220px + margins)
  const tooltipStyle: React.CSSProperties = {
    position: 'fixed',
    left: 232,
    top: Math.max(10, Math.min(window.innerHeight - 340, rect.top - 20)),
    width: 280,
    maxHeight: 480,
    background: 'var(--c-surface-container-highest, #1e1e2e)',
    border: '1.5px solid var(--node-color, var(--c-panel-border))',
    borderColor: entry.color,
    borderRadius: 8,
    padding: 12,
    boxShadow: '0 8px 32px rgba(0,0,0,0.55), 0 2px 8px rgba(0,0,0,0.3)',
    zIndex: 10000,
    overflowY: 'auto',
    fontFamily: 'Inter, sans-serif',
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
    pointerEvents: 'none', // Prevent capturing drag start or mouse clicks
  };

  return (
    <div style={tooltipStyle}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ color: entry.color, display: 'flex', alignItems: 'center' }}>
          <Icon name={entry.icon} size={14} />
        </span>
        <strong style={{ fontSize: 12, color: 'var(--c-on-surface)', fontWeight: 600 }}>{entry.label}</strong>
        <span style={{
          fontSize: 8,
          textTransform: 'uppercase',
          fontWeight: 750,
          background: 'rgba(255,255,255,0.06)',
          padding: '1px 5px',
          borderRadius: 3,
          color: entry.color,
          marginLeft: 'auto',
          letterSpacing: '0.04em'
        }}>
          {entry.family}
        </span>
      </div>

      <div style={{ fontSize: 10, color: 'var(--c-on-surface-variant)', lineHeight: 1.4, opacity: 0.9 }}>
        {entry.description}
      </div>

      {theory && (
        <>
          <div style={{ height: '0.5px', background: 'rgba(255,255,255,0.08)', margin: '4px 0' }} />
          <div>
            <div style={{ fontSize: 8, textTransform: 'uppercase', fontWeight: 700, color: entry.color, letterSpacing: '0.05em', marginBottom: 2 }}>
              Mathematical Engine
            </div>
            <div style={{
              background: 'var(--c-surface-container-lowest, #11111b)',
              padding: '4px 8px',
              borderRadius: 4,
              fontFamily: 'monospace',
              fontSize: 9,
              color: 'var(--c-on-surface)',
              overflowX: 'auto'
            }}>
              {theory.math}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ fontSize: 8, textTransform: 'uppercase', fontWeight: 700, color: 'var(--c-on-surface-variant)', letterSpacing: '0.05em' }}>
              Operational Workflow
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {theory.workflow.slice(0, 3).map((step: string, idx: number) => (
                <div key={idx} style={{ fontSize: 9, color: 'var(--c-on-surface-variant)', opacity: 0.85, lineHeight: 1.3 }}>
                  <span style={{ color: entry.color, fontWeight: 600 }}>{idx + 1}.</span> {step}
                </div>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 8, textTransform: 'uppercase', fontWeight: 700, color: 'var(--c-on-surface-variant)', letterSpacing: '0.05em', marginBottom: 2 }}>
              Usage Tip
            </div>
            <div style={{ fontSize: 10, color: 'var(--c-on-surface-variant)', fontStyle: 'italic', lineHeight: 1.4 }}>
              💡 {theory.tips[0]}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
