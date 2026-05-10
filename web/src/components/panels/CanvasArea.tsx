/**
 * Canvas area — three-zone layout:
 * 1. Visualization (data zone + optional process zone, stacked vertically)
 * 2. Standardized metrics bar (bottom, 48px, always same structure)
 *
 * Floating info chip (top-left) and zoom controls (top-right) overlay the viz.
 */

import Icon from '../common/Icon';
import { usePlayground } from '../../store';

export default function CanvasArea() {
  const { model, params, datasetId, datasetParams, isTraining, stopTraining, liveMetrics, setLiveMetrics, resetVersion } = usePlayground();

  if (!model) {
    return (
      <div className="canvas" id="canvas-area">
        <div className="canvas__empty">Select a model to begin</div>
      </div>
    );
  }

  const Viz = model.VisualizationComponent;

  return (
    <div className="canvas" id="canvas-area">
      {/* Top overlay */}
      <div className="canvas__top-overlay">
        <div className="canvas__info-chip">
          <span className="canvas__dot" />
          <span className="canvas__info-model">{model.shortName}</span>
          <span className="canvas__info-sep">·</span>
          <span className="canvas__info-detail">{model.vizLabel}</span>
        </div>
        <div className="canvas__zoom-controls">
          <button className="canvas__zoom-btn" title="Zoom in" id="btn-zoom-in">
            <Icon name="zoom_in" size={14} />
          </button>
          <button className="canvas__zoom-btn" title="Zoom out" id="btn-zoom-out">
            <Icon name="zoom_out" size={14} />
          </button>
          <button className="canvas__zoom-btn" title="Reset view" id="btn-reset-view">
            <Icon name="center_focus_strong" size={14} />
          </button>
        </div>
      </div>

      {/* Visualization — model component handles internal layout */}
      <div className="canvas__visualization">
        <Viz
          params={params}
          dataset={datasetId}
          datasetParams={datasetParams}
          isTraining={isTraining}
          resetVersion={resetVersion}
          onTrainingComplete={stopTraining}
          onMetricsUpdate={setLiveMetrics}
        />
      </div>

      {/* Standardized Metrics Bar — always 48px, always at bottom */}
      <div className="canvas__metrics-bar">
        {liveMetrics.map((m) => (
          <div key={m.label} className="canvas__metric-slot">
            <span className="canvas__metric-label">{m.label}</span>
            <span className={`canvas__metric-value ${m.isPrimary ? 'canvas__metric-value--primary' : ''}`}>
              {m.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
