/**
 * ControlsPanel — dynamically renders controls based on the active model's
 * parameter schema and the current mode (basic/advanced).
 * Shows Train/Reset buttons for trainable models.
 */

import { useState } from 'react';
import Icon from '../common/Icon';
import ConfirmDialog from '../common/ConfirmDialog';
import { usePlayground } from '../../store';
import type { ParamDescriptor } from '../../models';

/* ─── Individual param renderers ─── */

function SliderControl({
  param, value, onChange,
}: {
  param: Extract<ParamDescriptor, { type: 'slider' }>;
  value: number;
  onChange: (v: number) => void;
}) {
  const display = param.formatValue ? param.formatValue(value) : String(value);
  return (
    <div className="control__slider">
      <div className="control__row">
        <label className="control__label">{param.label}</label>
        <span className="control__value control__value--accent">{display}</span>
      </div>
      <input
        type="range"
        className="control__range"
        min={param.min}
        max={param.max}
        step={param.step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="control__range-labels">
        <span>{param.min}</span>
        <span>{param.max}</span>
      </div>
    </div>
  );
}

function SelectControl({
  param, value, onChange,
}: {
  param: Extract<ParamDescriptor, { type: 'select' }>;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="control__select-wrap">
      <label className="control__label">{param.label}</label>
      <div className="control__select-container">
        <select
          className="control__select"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {param.options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <Icon name="expand_more" size={14} className="control__select-icon" />
      </div>
    </div>
  );
}

function ToggleControl({
  param, value, onChange,
}: {
  param: Extract<ParamDescriptor, { type: 'toggle' }>;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="control__toggle">
      <span className="control__toggle-label">{param.label}</span>
      <div
        className={`control__switch ${value ? 'control__switch--on' : ''}`}
        onClick={() => onChange(!value)}
      >
        <span className="control__switch-thumb" />
      </div>
    </label>
  );
}

function ChipSelectControl({
  param, value, onChange,
}: {
  param: Extract<ParamDescriptor, { type: 'chip-select' }>;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="control__chips-wrap">
      <label className="control__label">{param.label}</label>
      <div className="control__chips">
        {param.options.map((opt) => (
          <div
            key={opt.value}
            className={`control__chip ${opt.value === value ? 'control__chip--active' : ''}`}
            onClick={() => onChange(opt.value)}
          >
            {opt.icon && <Icon name={opt.icon} size={10} />}
            {opt.label}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Generic param renderer ─── */
function ParamControl({
  param, value, onChange,
}: {
  param: ParamDescriptor;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  switch (param.type) {
    case 'slider':
      return <SliderControl param={param} value={value as number} onChange={onChange} />;
    case 'select':
      return <SelectControl param={param} value={value as string} onChange={onChange} />;
    case 'toggle':
      return <ToggleControl param={param} value={value as boolean} onChange={onChange} />;
    case 'chip-select':
      return <ChipSelectControl param={param} value={value as string} onChange={onChange} />;
    default:
      return null;
  }
}

/* ─── Main Panel ─── */
export default function ControlsPanel() {
  const {
    model, params, setParam, mode,
    datasetId, setDataset, datasetParams, setDatasetParam,
    isTraining, startTraining, stopTraining, resetTraining,
  } = usePlayground();
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  if (!model) return null;

  // Filter params by mode: basic shows only basic, advanced shows all
  const visibleParams = model.params.filter((p) => {
    if (mode === 'basic') return !p.level || p.level === 'basic';
    return true; // advanced shows everything
  });

  const toggleParams = visibleParams.filter((p) => p.type === 'toggle');
  const nonToggleParams = visibleParams.filter((p) => p.type !== 'toggle');

  // Split non-toggle params into basic and advanced groups for visual separation
  const basicNonToggle = nonToggleParams.filter((p) => !p.level || p.level === 'basic');
  const advancedNonToggle = nonToggleParams.filter((p) => p.level === 'advanced');
  const basicToggles = toggleParams.filter((p) => !p.level || p.level === 'basic');
  const advancedToggles = toggleParams.filter((p) => p.level === 'advanced');

  return (
    <aside className="controls" id="controls-panel">
      <div className="controls__body">
        {/* Parameters Section */}
        <section className="controls__section">
          <header className="controls__section-header">
            <span className="controls__section-title">Parameters</span>
            <span className="controls__section-badge">
              {mode === 'advanced' ? 'ADV' : 'STD'}
            </span>
          </header>

          {basicNonToggle.map((p) => (
            <ParamControl
              key={p.key}
              param={p}
              value={params[p.key]}
              onChange={(v) => setParam(p.key, v)}
            />
          ))}

          {basicToggles.length > 0 && (
            <div className="controls__toggles">
              {basicToggles.map((p) => (
                <ParamControl
                  key={p.key}
                  param={p}
                  value={params[p.key]}
                  onChange={(v) => setParam(p.key, v)}
                />
              ))}
            </div>
          )}
        </section>

        {/* Advanced Parameters Section — only when mode is advanced */}
        {mode === 'advanced' && advancedNonToggle.length + advancedToggles.length > 0 && (
          <section className="controls__section controls__section--advanced">
            <header className="controls__section-header">
              <span className="controls__section-title">Advanced</span>
              <Icon name="science" size={12} className="controls__section-icon" />
            </header>

            {advancedNonToggle.map((p) => (
              <ParamControl
                key={p.key}
                param={p}
                value={params[p.key]}
                onChange={(v) => setParam(p.key, v)}
              />
            ))}

            {advancedToggles.length > 0 && (
              <div className="controls__toggles">
                {advancedToggles.map((p) => (
                  <ParamControl
                    key={p.key}
                    param={p}
                    value={params[p.key]}
                    onChange={(v) => setParam(p.key, v)}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {/* Dataset Section */}
        <section className="controls__section">
          <header className="controls__section-header">
            <span className="controls__section-title">Dataset</span>
          </header>

          <div className="control__chips">
            {model.dataset.options.map((opt) => (
              <div
                key={opt.value}
                className={`control__chip ${opt.value === datasetId ? 'control__chip--active' : ''}`}
                onClick={() => setDataset(opt.value)}
              >
                {opt.icon && <Icon name={opt.icon} size={10} />}
                {opt.label}
              </div>
            ))}
          </div>

          {model.dataset.params.map((p) => (
            <ParamControl
              key={p.key}
              param={p}
              value={datasetParams[p.key]}
              onChange={(v) => setDatasetParam(p.key, v)}
            />
          ))}
        </section>
      </div>

      {/* Footer — Train/Reset for trainable models, or Explain button */}
      <div className="controls__footer">
        {model.trainable ? (
          <div className="controls__train-actions">
            <button
              className={`controls__train-btn ${isTraining ? 'controls__train-btn--stop' : ''}`}
              onClick={isTraining ? stopTraining : startTraining}
              id="btn-train"
            >
              <Icon name={isTraining ? 'stop' : 'play_arrow'} size={14} />
              <span>{isTraining ? 'Stop' : 'Train'}</span>
            </button>
            <button
              className="controls__reset-btn"
              onClick={() => setShowResetConfirm(true)}
              disabled={isTraining}
              id="btn-reset"
            >
              <Icon name="refresh" size={14} />
            </button>
          </div>
        ) : (
          <button className="controls__explain-btn" id="btn-explain">
            <Icon name="auto_awesome" size={14} />
            <span>Explain this</span>
          </button>
        )}
      </div>

      {/* Reset Confirmation Dialog */}
      <ConfirmDialog
        open={showResetConfirm}
        title="Reset Model"
        message="This will reset all weights, loss history, and parameters to their defaults. Are you sure?"
        confirmLabel="Reset"
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={() => { setShowResetConfirm(false); resetTraining(); }}
        onCancel={() => setShowResetConfirm(false)}
      />
    </aside>
  );
}
