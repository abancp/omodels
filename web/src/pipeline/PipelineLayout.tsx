/**
 * PipelineLayout — main layout for the Pipeline builder page
 */

import { useState, useRef } from 'react';
import { PipelineProvider, usePipeline } from './PipelineStore';
import NodePalette from './components/NodePalette';
import PipelineCanvas from './components/PipelineCanvas';
import NodeInspector from './components/NodeInspector';
import ExecutionLog from './components/ExecutionLog';
import type { ViewType } from '../components/layout/TopNavBar';
import TopNavBar from '../components/layout/TopNavBar';
import SearchOverlay from '../components/common/SearchOverlay';
import Icon from '../components/common/Icon';

interface PipelineLayoutProps {
  activeView: ViewType;
  onViewChange: (v: ViewType) => void;
}

function PipelineWorkspace({ activeView, onViewChange }: PipelineLayoutProps) {
  const { pipeline, runPipeline, resetPipeline, savePipeline, loadPipeline, isRunning, renamePipeline } = usePipeline();
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [logOpen, setLogOpen] = useState(true);
  const [editingName, setEditingName] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleLoadFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => { try { loadPipeline(ev.target?.result as string); } catch { alert('Invalid pipeline file.'); } };
    reader.readAsText(file);
  };

  const stateColor = {
    draft: 'var(--c-on-surface-variant)',
    validating: '#f59e0b',
    training: '#f59e0b',
    trained: '#10b981',
    stale: '#f59e0b',
    error: '#ef4444',
  }[pipeline.state] ?? 'var(--c-on-surface-variant)';

  return (
    <div className="app" id="pipeline-root">
      <TopNavBar onSearchClick={() => setIsSearchOpen(p => !p)} activeView={activeView} onViewChange={onViewChange} />

      {/* Toolbar */}
      <div className="pipeline-toolbar">
        <div className="pipeline-toolbar__left">
          {editingName ? (
            <input
              ref={nameRef}
              className="pipeline-toolbar__name-input"
              defaultValue={pipeline.name}
              autoFocus
              onBlur={e => { renamePipeline(e.target.value || 'Untitled Pipeline'); setEditingName(false); }}
              onKeyDown={e => { if (e.key === 'Enter') e.currentTarget.blur(); }}
            />
          ) : (
            <span className="pipeline-toolbar__name" onClick={() => setEditingName(true)} title="Click to rename">
              {pipeline.name}
              <Icon name="edit" size={12} style={{ marginLeft: 6, opacity: 0.4 }} />
            </span>
          )}
          <div className="pipeline-toolbar__state" style={{ color: stateColor }}>
            ● {pipeline.state.charAt(0).toUpperCase() + pipeline.state.slice(1)}
          </div>
        </div>

        <div className="pipeline-toolbar__actions">
          <button
            className="pipeline-toolbar__btn"
            onClick={() => fileRef.current?.click()}
            title="Load pipeline"
          >
            <Icon name="folder_open" size={14} />
            <span>Load</span>
          </button>
          <input ref={fileRef} type="file" accept=".om,.json" onChange={handleLoadFile} style={{ display: 'none' }} />

          <button className="pipeline-toolbar__btn" onClick={savePipeline} title="Save pipeline">
            <Icon name="save" size={14} />
            <span>Save</span>
          </button>

          <button className="pipeline-toolbar__btn" onClick={resetPipeline} title="Reset training" disabled={isRunning}>
            <Icon name="refresh" size={14} />
            <span>Reset</span>
          </button>

          <button
            className={`pipeline-toolbar__btn pipeline-toolbar__btn--run ${isRunning ? 'pipeline-toolbar__btn--running' : ''}`}
            onClick={runPipeline}
            disabled={isRunning}
            title="Run pipeline"
          >
            {isRunning ? (
              <><Icon name="hourglass_empty" size={14} /><span>Running…</span></>
            ) : (
              <><Icon name="play_arrow" size={14} /><span>Run Pipeline</span></>
            )}
          </button>
        </div>
      </div>

      {/* Main 3-column workspace */}
      <div className="pipeline-workspace">
        {/* Left: Node Palette */}
        <NodePalette />

        {/* Center: Canvas + Log */}
        <div className="pipeline-center">
          <PipelineCanvas />
          {logOpen && (
            <div className="pipeline-log-slot">
              <ExecutionLog />
            </div>
          )}
          <button
            className="pipeline-log-toggle"
            onClick={() => setLogOpen(p => !p)}
            title={logOpen ? 'Hide log' : 'Show log'}
          >
            <Icon name={logOpen ? 'expand_more' : 'expand_less'} size={14} />
            {logOpen ? 'Hide Log' : 'Show Log'}
          </button>
        </div>

        {/* Right: Inspector */}
        <NodeInspector />
      </div>

      <SearchOverlay isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} />
    </div>
  );
}

export default function PipelineLayout(props: PipelineLayoutProps) {
  return (
    <PipelineProvider>
      <PipelineWorkspace {...props} />
    </PipelineProvider>
  );
}
