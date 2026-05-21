/**
 * ControlsPanel — dynamically renders controls based on the active model's
 * parameter schema and the current mode (basic/advanced).
 * Shows Train/Reset buttons for trainable models.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Icon from '../common/Icon';
import ConfirmDialog from '../common/ConfirmDialog';
import { usePlayground } from '../../store';
import type { ParamDescriptor } from '../../models';
import { PlaygroundDataEngine } from '../../models/dataEngine';

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

interface ImportScatterPlotProps {
  data: any[];
  modelType: 'regression' | 'classification' | 'clustering';
  isMultiInput: boolean;
  importMapping: { x: string; y: string; label?: string; features?: string[] };
}

function ImportScatterPlot({ data, modelType, isMultiInput, importMapping }: ImportScatterPlotProps) {
  const [hoveredPoint, setHoveredPoint] = React.useState<any | null>(null);

  // Filter out any points with invalid values
  const validPoints = React.useMemo(() => {
    if (!data) return [];
    return data.filter(p => isFinite(p.x) && isFinite(p.y));
  }, [data]);

  // Compute boundaries for scaling
  const bounds = React.useMemo(() => {
    if (validPoints.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    const xs = validPoints.map(p => p.x);
    const ys = validPoints.map(p => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);

    // Add buffer
    const xRange = maxX - minX || 1;
    const yRange = maxY - minY || 1;
    return {
      minX: minX - xRange * 0.08,
      maxX: maxX + xRange * 0.08,
      minY: minY - yRange * 0.08,
      maxY: maxY + yRange * 0.08,
    };
  }, [validPoints]);

  const svgWidth = 500;
  const svgHeight = 280;
  const paddingLeft = 45;
  const paddingRight = 15;
  const paddingTop = 15;
  const paddingBottom = 35;

  const plotWidth = svgWidth - paddingLeft - paddingRight;
  const plotHeight = svgHeight - paddingTop - paddingBottom;

  const mapX = (x: number) => {
    const range = bounds.maxX - bounds.minX;
    return paddingLeft + ((x - bounds.minX) / (range || 1)) * plotWidth;
  };

  const mapY = (y: number) => {
    const range = bounds.maxY - bounds.minY;
    return paddingTop + plotHeight - ((y - bounds.minY) / (range || 1)) * plotHeight;
  };

  // Grid lines fractions
  const gridFractions = [0.25, 0.5, 0.75];

  // Helper for colors
  const getPointColor = (p: any) => {
    if (modelType === 'classification') {
      const cls = p.cls ?? 0;
      const colors = ['#cfbcff', '#e7c365', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#14b8a6'];
      return colors[cls % colors.length];
    } else if (modelType === 'regression') {
      const val = p.label ?? p.y;
      const targetMin = bounds.minY;
      const targetMax = bounds.maxY;
      const pct = (val - targetMin) / (targetMax - targetMin || 1);
      const hue = 190 + Math.max(0, Math.min(1, pct)) * 150;
      return `hsl(${hue}, 85%, 70%)`;
    } else {
      return '#22d3ee';
    }
  };

  // Unique classes for classification legend
  const uniqueClasses = React.useMemo(() => {
    if (modelType !== 'classification') return [];
    return [...new Set(validPoints.map(p => p.cls ?? 0))].sort((a, b) => a - b);
  }, [validPoints, modelType]);

  const xLabel = isMultiInput ? `Feature 0 (${importMapping.features?.[0] || 'F0'})` : importMapping.x;
  const yLabel = isMultiInput
    ? (modelType === 'regression' ? `Target (${importMapping.label || 'Y'})` : `Feature 1 (${importMapping.features?.[1] || 'F1'})`)
    : importMapping.y;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 12px', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: 10 }}>
        <div style={{ display: 'flex', gap: 12 }}>
          {modelType === 'classification' && (
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {uniqueClasses.map(cls => {
                const colors = ['#cfbcff', '#e7c365', '#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#14b8a6'];
                return (
                  <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: colors[cls % colors.length] }} />
                    <span style={{ opacity: 0.8 }}>Class {cls}</span>
                  </div>
                );
              })}
            </div>
          )}
          {modelType === 'regression' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ opacity: 0.8 }}>Target value:</span>
              <div style={{ width: 80, height: 8, borderRadius: 4, background: 'linear-gradient(to right, hsl(190, 85%, 70%), hsl(340, 85%, 70%))' }} />
              <span style={{ fontSize: 9, opacity: 0.5 }}>Min → Max</span>
            </div>
          )}
          {modelType === 'clustering' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#22d3ee' }} />
              <span style={{ opacity: 0.8 }}>Data Points</span>
            </div>
          )}
        </div>
        <div>
          {hoveredPoint && (
            <span style={{ fontFamily: "'JetBrains Mono', monospace", color: 'var(--c-primary)' }}>
              Hovered: ({hoveredPoint.x.toFixed(3)}, {hoveredPoint.y.toFixed(3)})
              {modelType === 'classification' && ` · Class: ${hoveredPoint.cls}`}
              {modelType === 'regression' && ` · Val: ${(hoveredPoint.label ?? hoveredPoint.y).toFixed(3)}`}
            </span>
          )}
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0, position: 'relative', background: '#0d0d16', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
        {validPoints.length === 0 ? (
          <div style={{ color: 'rgba(255,255,255,0.3)', fontSize: 11 }}>Please select valid X and Y features to display the plot.</div>
        ) : (
          <svg width="100%" height="100%" viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="xMidYMid meet" style={{ background: 'transparent' }}>
            {/* Grid Y lines */}
            {gridFractions.map((g, idx) => {
              const yVal = bounds.minY + g * (bounds.maxY - bounds.minY);
              const sy = mapY(yVal);
              return (
                <g key={`gy-${idx}`}>
                  <line x1={paddingLeft} y1={sy} x2={svgWidth - paddingRight} y2={sy} stroke="rgba(255,255,255,0.04)" strokeDasharray="3,3" strokeWidth="0.8" />
                  <text x={paddingLeft - 6} y={sy + 3} fill="rgba(255,255,255,0.35)" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="end">
                    {yVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Grid X lines */}
            {gridFractions.map((g, idx) => {
              const xVal = bounds.minX + g * (bounds.maxX - bounds.minX);
              const sx = mapX(xVal);
              return (
                <g key={`gx-${idx}`}>
                  <line x1={sx} y1={paddingTop} x2={sx} y2={svgHeight - paddingBottom} stroke="rgba(255,255,255,0.04)" strokeDasharray="3,3" strokeWidth="0.8" />
                  <text x={sx} y={svgHeight - paddingBottom + 12} fill="rgba(255,255,255,0.35)" fontSize="8" fontFamily="'JetBrains Mono', monospace" textAnchor="middle">
                    {xVal.toFixed(1)}
                  </text>
                </g>
              );
            })}

            {/* Axis Lines */}
            <line x1={paddingLeft} y1={paddingTop} x2={paddingLeft} y2={svgHeight - paddingBottom} stroke="rgba(255,255,255,0.12)" strokeWidth="1" />
            <line x1={paddingLeft} y1={svgHeight - paddingBottom} x2={svgWidth - paddingRight} y2={svgHeight - paddingBottom} stroke="rgba(255,255,255,0.12)" strokeWidth="1" />

            {/* Axis Labels */}
            <text x={paddingLeft + plotWidth / 2} y={svgHeight - 6} fill="rgba(255,255,255,0.6)" fontSize="9" fontWeight="600" textAnchor="middle" letterSpacing="0.02em">
              {xLabel}
            </text>
            <text x={10} y={paddingTop + plotHeight / 2} fill="rgba(255,255,255,0.6)" fontSize="9" fontWeight="600" textAnchor="middle" transform={`rotate(-90, 10, ${paddingTop + plotHeight / 2})`} letterSpacing="0.02em">
              {yLabel}
            </text>

            {/* Scatter points */}
            {validPoints.map((p, idx) => {
              const cx = mapX(p.x);
              const cy = mapY(p.y);
              const color = getPointColor(p);
              const isHovered = hoveredPoint === p;
              return (
                <g key={idx}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isHovered ? 8 : 4.5}
                    fill={color}
                    fillOpacity={isHovered ? 0.35 : 0.75}
                    stroke={color}
                    strokeWidth={isHovered ? 2 : 0}
                    style={{ transition: 'r 0.15s, fill-opacity 0.15s' }}
                    onMouseEnter={() => setHoveredPoint(p)}
                    onMouseLeave={() => setHoveredPoint(null)}
                    cursor="pointer"
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r={2}
                    fill="#ffffff"
                    fillOpacity={isHovered ? 0.9 : 0}
                    pointerEvents="none"
                  />
                </g>
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
}

/* ─── Main Panel ─── */
export default function ControlsPanel() {
  const {
    model, params, setParam, mode, activeModelId,
    datasetId, setDataset, datasetParams, setDatasetParam,
    isTraining, startTraining, stopTraining, resetTraining,
    setImportedData, setImportStats, importStats, testData, setTestData, testResults,
    saveModel, loadModel,
  } = usePlayground();
  const [showResetConfirm, setShowResetConfirm] = useState(false);

  // Import state
  const fileInputRef = useRef<HTMLInputElement>(null);
  const omFileInputRef = useRef<HTMLInputElement>(null);
  const [showImportPopup, setShowImportPopup] = useState(false);
  const [importResult, setImportResult] = useState<import('../../models/dataEngine').ParseResult | null>(null);

  const handleOMUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        loadModel(text);
      } catch (err: any) {
        alert(err.message || 'Failed to load model file.');
      }
    };
    reader.readAsText(file);
    e.target.value = ''; // Reset file input
  }, [loadModel]);
  const [importMapping, setImportMapping] = useState<{
    x: string;
    y: string;
    label?: string;
    features?: string[];
  }>({ x: '', y: '', features: [] });
  const [importValidation, setImportValidation] = useState<import('../../models/dataEngine').ValidationResult | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [autoScale, setAutoScale] = useState(false);
  const [autoSplit, setAutoSplit] = useState(true);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [previewTab, setPreviewTab] = useState<'raw' | 'vector' | 'plot'>('plot');
  const prevDatasetRef = useRef(datasetId);
  // Categorical encoding: maps column name -> 'one-hot' | 'label' | 'skip'
  const [catEncodings, setCatEncodings] = useState<Record<string, 'one-hot' | 'label' | 'skip'>>({});

  // Test dataset state
  const testFileRef = useRef<HTMLInputElement>(null);
  // const [showTestPopup, setShowTestPopup] = useState(false);
  const [showTestResults, setShowTestResults] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);

  // Keep importMapping.features and validation in sync with inputNodes parameter
  const isMultiInput = ['mlp', 'perceptron'].includes(activeModelId);
  const numInputs = isMultiInput ? ((params.inputNodes as number) ?? 2) : 2;

  // Model-specific configuration for data mapping
  const modelConfig = React.useMemo(() => {
    const cfg = {
      title: 'Dataset Import',
      description: 'Configure your custom CSV/JSON columns.',
      helpText: 'Select columns to map to the model features.',
      xLabel: 'Feature 1 (X Axis)',
      yLabel: 'Feature 2 (Y Axis)',
      targetLabel: 'Label / Class Column',
      showTarget: false,
      inputCount: 2,
      details: ''
    };

    if (activeModelId === 'linear-regression' || activeModelId === 'polynomial-regression') {
      cfg.title = 'Simple Regression Setup';
      cfg.description = 'Map columns for 2D Simple Regression analysis.';
      cfg.helpText = 'Regression fits a line mapping 1 Independent input feature to a continuous Dependent output.';
      cfg.xLabel = 'Independent Variable (X Axis)';
      cfg.yLabel = 'Target Variable (Y Axis)';
      cfg.showTarget = false;
      cfg.inputCount = 1;
      cfg.details = 'Requires exactly one numeric feature for X, and one numeric target variable for Y.';
    } else if (['logistic-regression', 'svm', 'naive-bayes', 'knn', 'decision-tree', 'random-forest', 'gradient-boosting'].includes(activeModelId)) {
      cfg.title = '2D Classification Setup';
      cfg.description = 'Map columns to coordinate features and discrete categories.';
      cfg.helpText = '2D Classifiers map coordinates (X, Y) to a categorical class label for boundary drawing.';
      cfg.xLabel = 'Feature 1 (X Coordinate)';
      cfg.yLabel = 'Feature 2 (Y Coordinate)';
      cfg.targetLabel = 'Label / Class Column';
      cfg.showTarget = true;
      cfg.inputCount = 2;
      cfg.details = 'Requires two numeric coordinates (X, Y) and a discrete category column (numeric or text) for class coloring.';
    } else if (['kmeans', 'dbscan', 'gmm'].includes(activeModelId)) {
      cfg.title = '2D Clustering Setup';
      cfg.description = 'Map columns to spatial features for grouping.';
      cfg.helpText = 'Unsupervised clustering groups items in 2D coordinate space. No labels needed.';
      cfg.xLabel = 'Feature 1 (X Coordinate)';
      cfg.yLabel = 'Feature 2 (Y Coordinate)';
      cfg.showTarget = false;
      cfg.inputCount = 2;
      cfg.details = 'Requires two numeric spatial coordinates (X, Y) to discover cluster groupings.';
    } else if (['perceptron', 'mlp'].includes(activeModelId)) {
      const modelType = PlaygroundDataEngine.getModelDataType(activeModelId, params);
      cfg.title = modelType === 'regression' ? 'Deep Neural Regression Setup' : 'Deep Neural Classification Setup';
      cfg.description = `Map multi-dimensional features for Neural Network analysis (${numInputs} inputs).`;
      cfg.helpText = `Feed-forward networks map ${numInputs} numeric inputs to a target column.`;
      cfg.targetLabel = modelType === 'regression' ? 'Continuous Target Variable (Y)' : 'Label / Class Column';
      cfg.showTarget = true;
      cfg.inputCount = numInputs;
      cfg.details = `Maps exactly ${numInputs} features (numeric or encoded) to a ${modelType === 'regression' ? 'continuous target value' : 'discrete label class'}.`;
    }

    return cfg;
  }, [activeModelId, params, numInputs]);

  // Compute available feature columns including encoded categoricals
  const availableFeatureCols = React.useMemo(() => {
    if (!importResult) return [];
    const cols = [...importResult.numericColumns];
    for (const [col, encoding] of Object.entries(catEncodings)) {
      if (encoding === 'skip' || col === importMapping.label) continue;
      if (encoding === 'one-hot') {
        const categories = [...new Set(importResult.data.map(r => String(r[col] ?? '')))].sort();
        for (const cat of categories) cols.push(`${col}_${cat}`);
      } else if (encoding === 'label') {
        cols.push(col);
      }
    }
    return cols;
  }, [importResult, catEncodings, importMapping.label]);

  useEffect(() => {
    if (!importResult) return;
    setImportMapping(prev => {
      const validCols = availableFeatureCols.filter(c => c !== prev.label);
      const newFeatures = Array.from({ length: numInputs }).map((_, i) => {
        const existing = prev.features?.[i];
        if (existing && validCols.includes(existing)) return existing;
        return validCols[Math.min(i, validCols.length - 1)] || '';
      });
      return {
        ...prev,
        features: newFeatures,
      };
    });
  }, [importResult, activeModelId, availableFeatureCols, numInputs]);

  useEffect(() => {
    if (!importResult) return;

    let isValid = false;
    let errorMsg: string | undefined = undefined;
    const totalAvailable = availableFeatureCols.length;

    if (isMultiInput) {
      const selectedFeatures = (importMapping.features || []).filter(c => c !== importMapping.label);
      const mappedFeaturesCount = selectedFeatures.filter(f => f).length;

      if (mappedFeaturesCount < numInputs) {
        errorMsg = `Please map all ${numInputs} input features.`;
      } else if (!importMapping.label) {
        errorMsg = `Please select a target/label column.`;
      } else {
        isValid = true;
      }
    } else {
      // 2D Models
      if (!importMapping.x || !importMapping.y) {
        errorMsg = 'Please map both X and Y columns.';
      } else if (importMapping.x === importMapping.y) {
        errorMsg = 'X Axis and Y Axis columns must be different.';
      } else if (modelConfig.showTarget && !importMapping.label) {
        errorMsg = 'Please select a Class / Label column.';
      } else if (modelConfig.showTarget && (importMapping.x === importMapping.label || importMapping.y === importMapping.label)) {
        errorMsg = 'Label column cannot be the same as Feature coordinates (X or Y).';
      } else {
        isValid = true;
      }
    }

    setImportValidation({
      valid: isValid,
      info: `${importResult.data.length} rows · ${totalAvailable} features available · mapping ${isMultiInput ? `${numInputs} inputs` : '2 coordinates'}`,
      error: errorMsg
    });
  }, [
    importResult,
    activeModelId,
    isMultiInput,
    availableFeatureCols,
    importMapping.features,
    importMapping.x,
    importMapping.y,
    importMapping.label,
    numInputs,
    modelConfig.showTarget
  ]);

  // Compute live preview data
  const previewData = React.useMemo(() => {
    if (!importResult || !importValidation?.valid) return null;
    try {
      const modelType = PlaygroundDataEngine.getModelDataType(activeModelId, params);
      let workingData = importResult.data;
      for (const [col, encoding] of Object.entries(catEncodings)) {
        if (encoding === 'skip' || col === importMapping.label) continue;
        if (encoding === 'one-hot') workingData = PlaygroundDataEngine.oneHotEncode(workingData, col).data;
        else if (encoding === 'label') workingData = PlaygroundDataEngine.labelEncode(workingData, col).data;
      }

      let pts: any[] = [];
      if (isMultiInput) {
        const featureCols = (importMapping.features || []).filter(f => f !== importMapping.label);
        let labelMap: Map<any, number> | null = null;
        if (modelType === 'classification') {
          const uniqueLabels = [...new Set(workingData.map(r => r[importMapping.label!]))].filter(v => v != null);
          labelMap = new Map(uniqueLabels.map((l, i) => [l, i]));
        }
        pts = workingData.map(r => {
          const features = featureCols.map(col => Number(r[col]));
          let target = 0;
          if (modelType === 'classification') target = labelMap!.get(r[importMapping.label!]) ?? 0;
          else target = Number(r[importMapping.label!]);
          return {
            features, label: target,
            x: features[0] ?? 0, y: features[1] ?? 0, cls: target,
          };
        }).filter(p => p.features.every(isFinite) && isFinite(p.label));
      } else {
        if (modelType === 'regression') {
          pts = PlaygroundDataEngine.toRegressionPoints(workingData, importMapping);
        } else if (modelType === 'classification') {
          pts = PlaygroundDataEngine.toClassificationPoints(workingData, importMapping);
        } else if (modelType === 'clustering') {
          pts = PlaygroundDataEngine.toClusteringPoints(workingData, importMapping);
        }
      }
      if (pts.length < 2) return null;
      if (autoScale) {
        if (isMultiInput && pts.length > 0) {
          const dim = pts[0].features.length;
          const mins = new Array(dim).fill(Infinity);
          const maxs = new Array(dim).fill(-Infinity);
          let targetMin = Infinity;
          let targetMax = -Infinity;
          for (const p of pts) {
            for (let j = 0; j < dim; j++) {
              if (p.features[j] < mins[j]) mins[j] = p.features[j];
              if (p.features[j] > maxs[j]) maxs[j] = p.features[j];
            }
            if (modelType === 'regression') {
              if (p.label < targetMin) targetMin = p.label;
              if (p.label > targetMax) targetMax = p.label;
            }
          }
          pts = pts.map(p => {
            const norm = p.features.map((v: number, j: number) => {
              const r = maxs[j] - mins[j];
              return r === 0 ? 0.5 : (v - mins[j]) / r;
            });
            let normLabel = p.label;
            if (modelType === 'regression') {
              const r = targetMax - targetMin;
              normLabel = r === 0 ? 0.5 : (p.label - targetMin) / r;
            }
            return { ...p, features: norm, x: norm[0] ?? 0.5, y: norm[1] ?? 0.5, label: normLabel, cls: modelType === 'regression' ? normLabel : p.cls };
          });
        } else if (modelType !== 'regression') {
          pts = PlaygroundDataEngine.normalizePoints(pts).points;
        }
      }
      return pts.slice(0, 300);
    } catch {
      return null;
    }
  }, [importResult, importValidation?.valid, activeModelId, importMapping, catEncodings, autoScale, isMultiInput]);

  // When dataset changes to 'import', open file picker
  useEffect(() => {
    if (datasetId === 'import' && prevDatasetRef.current !== 'import') {
      setImportedData(null);
      setTimeout(() => fileInputRef.current?.click(), 80);
    }
    prevDatasetRef.current = datasetId;
  }, [datasetId, setImportedData]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setImportError(null);
      const result = await PlaygroundDataEngine.parseFile(file);
      const modelType = PlaygroundDataEngine.getModelDataType(activeModelId, params);
      const validation = PlaygroundDataEngine.validate(result, modelType);
      const mapping = PlaygroundDataEngine.suggestMapping(result, modelType);

      // Default features: all numeric columns
      const initialFeatures = result.numericColumns;

      // Initialize categorical encoding defaults based on cardinality
      const defaultEncodings: Record<string, 'one-hot' | 'label' | 'skip'> = {};
      for (const col of result.categoricalColumns) {
        let uniqueCount = 0;
        const seen = new Set();
        for (const row of result.data) {
          const val = row[col];
          if (val != null && val !== '') {
            seen.add(val);
            if (seen.size > 15) break;
          }
        }
        uniqueCount = seen.size;

        if (uniqueCount > 0 && uniqueCount <= 15) {
          defaultEncodings[col] = 'one-hot';
        } else if (uniqueCount > 15) {
          defaultEncodings[col] = 'label';
        } else {
          defaultEncodings[col] = 'skip';
        }
      }
      setCatEncodings(defaultEncodings);

      setImportResult(result);
      setImportMapping({
        x: mapping.x,
        y: mapping.y,
        label: mapping.label,
        features: initialFeatures,
      });
      setImportValidation(validation);
      setShowImportPopup(true);
    } catch (err: any) {
      setImportError(err.message || 'Failed to parse file');
    }
  };

  const handleApplyImport = async () => {
    if (!importResult) return;
    const modelType = PlaygroundDataEngine.getModelDataType(activeModelId, params);

    if (!isMultiInput && importMapping.x === importMapping.y) {
      setImportError('X and Y must be different columns');
      return;
    }

    // Step 1: Apply categorical encodings to get an expanded dataset
    let workingData = importResult.data;
    let expandedNumericCols = [...importResult.numericColumns];

    for (const [col, encoding] of Object.entries(catEncodings)) {
      if (encoding === 'skip') continue;
      if (col === importMapping.label) continue; // Don't encode the label column
      if (encoding === 'one-hot') {
        const result = PlaygroundDataEngine.oneHotEncode(workingData, col);
        workingData = result.data;
        expandedNumericCols.push(...result.newCols);
      } else if (encoding === 'label') {
        const result = PlaygroundDataEngine.labelEncode(workingData, col);
        workingData = result.data;
        expandedNumericCols.push(col);
      }
    }

    let pts: any[] = [];
    if (isMultiInput) {
      if (!importMapping.label) {
        setImportError(`Please select a ${modelType === 'regression' ? 'target' : 'label'} column`);
        return;
      }
      const featureCols = (importMapping.features || []).filter(f => f !== importMapping.label);
      if (featureCols.length < 2) {
        setImportError('Please select at least 2 features');
        return;
      }
      let labelMap: Map<any, number> | null = null;
      if (modelType === 'classification') {
        const uniqueLabels = [...new Set(workingData.map(r => r[importMapping.label!]))].filter(v => v != null);
        labelMap = new Map(uniqueLabels.map((l, i) => [l, i]));
      }
      pts = workingData.map(r => {
        const features = featureCols.map(col => Number(r[col]));
        let target = 0;
        if (modelType === 'classification') target = labelMap!.get(r[importMapping.label!]) ?? 0;
        else target = Number(r[importMapping.label!]);
        return {
          features, label: target,
          x: features[0] ?? 0, y: features[1] ?? 0, cls: target,
        };
      }).filter(p => p.features.every(isFinite) && isFinite(p.label));
    } else {
      switch (modelType) {
        case 'regression':
          pts = PlaygroundDataEngine.toRegressionPoints(workingData, importMapping);
          break;
        case 'classification':
          if (!importMapping.label) {
            setImportError('Please select a label column');
            return;
          }
          pts = PlaygroundDataEngine.toClassificationPoints(workingData, importMapping);
          break;
        case 'clustering':
          pts = PlaygroundDataEngine.toClusteringPoints(workingData, importMapping);
          break;
      }
    }

    if (pts.length < 2) {
      setImportError('Need at least 2 valid data points after conversion');
      return;
    }

    // Feature normalization (if auto-scale is enabled)
    if (autoScale) {
      if (isMultiInput && pts.length > 0) {
        const dim = pts[0].features.length;
        const mins = new Array(dim).fill(Infinity);
        const maxs = new Array(dim).fill(-Infinity);
        let targetMin = Infinity;
        let targetMax = -Infinity;
        for (const p of pts) {
          for (let j = 0; j < dim; j++) {
            const v = p.features[j];
            if (v < mins[j]) mins[j] = v;
            if (v > maxs[j]) maxs[j] = v;
          }
          if (modelType === 'regression') {
            if (p.label < targetMin) targetMin = p.label;
            if (p.label > targetMax) targetMax = p.label;
          }
        }
        pts = pts.map(p => {
          const normFeatures = p.features.map((v: number, j: number) => {
            const range = maxs[j] - mins[j];
            return range === 0 ? 0.5 : (v - mins[j]) / range;
          });
          let normLabel = p.label;
          if (modelType === 'regression') {
            const range = targetMax - targetMin;
            normLabel = range === 0 ? 0.5 : (p.label - targetMin) / range;
          }
          return {
            ...p,
            features: normFeatures,
            x: normFeatures[0] ?? 0.5,
            y: normFeatures[1] ?? 0.5,
            label: normLabel,
            cls: modelType === 'regression' ? normLabel : p.cls,
          };
        });
        setImportStats({ mins, maxs, targetMin, targetMax });
      } else {
        const { points, stats } = PlaygroundDataEngine.normalizePoints(pts);
        pts = points;
        setImportStats({ mins: [stats.xMin, stats.yMin], maxs: [stats.xMax, stats.yMax] });
      }
    } else {
      setImportStats(null);
    }

    if (isMultiInput) {
      // Do not overwrite inputNodes, it's explicitly mapped
    }

    // Auto Split Train / Test
    if (autoSplit) {
      const shuffled = [...pts].sort(() => Math.random() - 0.5);
      const trainCount = Math.floor(shuffled.length * trainRatio);
      const trainPts = shuffled.slice(0, trainCount);
      const testPts = shuffled.slice(trainCount);

      if (trainPts.length < 2) {
        setImportError('Training set must have at least 2 samples after split');
        return;
      }

      setImportedData(trainPts);
      setTestData(testPts);
    } else {
      setImportedData(pts);
      setTestData([]);
    }

    setShowImportPopup(false);
    setImportError(null);
  };

  // ── Test dataset handler ──
  const handleTestFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      setTestError(null);
      const result = await PlaygroundDataEngine.parseFile(file);
      const modelType = PlaygroundDataEngine.getModelDataType(activeModelId, params);

      // Apply same categorical encodings
      let workingData = result.data;
      for (const [col, encoding] of Object.entries(catEncodings)) {
        if (encoding === 'skip' || col === importMapping.label) continue;
        if (encoding === 'one-hot') workingData = PlaygroundDataEngine.oneHotEncode(workingData, col).data;
        else if (encoding === 'label') workingData = PlaygroundDataEngine.labelEncode(workingData, col).data;
      }

      let pts: any[] = [];
      if (isMultiInput) {
        const featureCols = (importMapping.features || []).filter(f => f !== importMapping.label);
        let labelMap: Map<any, number> | null = null;
        if (modelType === 'classification') {
          const uniqueLabels = [...new Set(workingData.map(r => r[importMapping.label!]))].filter(v => v != null);
          labelMap = new Map(uniqueLabels.map((l, i) => [l, i]));
        }
        pts = workingData.map(r => {
          const features = featureCols.map(col => Number(r[col]));
          let target = 0;
          if (modelType === 'classification') target = labelMap!.get(r[importMapping.label!]) ?? 0;
          else target = Number(r[importMapping.label!]);
          return { features, label: target, x: features[0] ?? 0, y: features[1] ?? 0, cls: target };
        }).filter(p => p.features.every(isFinite) && isFinite(p.label));
      } else {
        if (modelType === 'regression') pts = PlaygroundDataEngine.toRegressionPoints(workingData, importMapping);
        else if (modelType === 'classification') pts = PlaygroundDataEngine.toClassificationPoints(workingData, importMapping);
        else pts = PlaygroundDataEngine.toClusteringPoints(workingData, importMapping);
      }

      if (pts.length < 2) { setTestError('Need at least 2 valid points'); return; }

      // Apply same scaling using training stats
      if (autoScale) {
        const stats = importStats as any;
        if (isMultiInput && pts[0]?.features && stats?.mins) {
          pts = pts.map(p => {
            const norm = p.features.map((v: number, j: number) => {
              const r = (stats.maxs[j] ?? 1) - (stats.mins[j] ?? 0);
              return r === 0 ? 0.5 : (v - (stats.mins[j] ?? 0)) / r;
            });
            let normLabel = p.label;
            if (modelType === 'regression' && stats.targetMin != null) {
              const r = stats.targetMax - stats.targetMin;
              normLabel = r === 0 ? 0.5 : (p.label - stats.targetMin) / r;
            }
            return { ...p, features: norm, x: norm[0] ?? 0.5, y: norm[1] ?? 0.5, label: normLabel };
          });
        } else if (stats) {
          const rx = (stats.maxs[0] ?? 1) - (stats.mins[0] ?? 0);
          const ry = (stats.maxs[1] ?? 1) - (stats.mins[1] ?? 0);
          pts = pts.map(p => {
            const nx = rx === 0 ? 0.5 : (p.x - (stats.mins[0] ?? 0)) / rx;
            const ny = ry === 0 ? 0.5 : (p.y - (stats.mins[1] ?? 0)) / ry;
            return { ...p, x: nx, y: ny };
          });
        }
      }

      setTestData(pts);
      // setShowTestPopup(false);
      setShowTestResults(true);
    } catch (err: any) {
      setTestError(err.message || 'Failed to parse test file');
    }
  };

  if (!model) return null;

  // Filter params by mode
  const visibleParams = model.params.filter((p) => {
    if (mode === 'basic') return !p.level || p.level === 'basic';
    return true;
  });

  const toggleParams = visibleParams.filter((p) => p.type === 'toggle');
  const nonToggleParams = visibleParams.filter((p) => p.type !== 'toggle');
  const basicNonToggle = nonToggleParams.filter((p) => !p.level || p.level === 'basic');
  const advancedNonToggle = nonToggleParams.filter((p) => p.level === 'advanced');
  const basicToggles = toggleParams.filter((p) => !p.level || p.level === 'basic');
  const advancedToggles = toggleParams.filter((p) => p.level === 'advanced');

  /* Inline styles for import popup */
  const overlayS: React.CSSProperties = {
    position: 'fixed', inset: 0, zIndex: 9999,
    background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(4px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  };
  const dialogS: React.CSSProperties = {
    background: 'var(--c-surface-container, #1e1e2e)', color: 'var(--c-on-surface, #e0e0e0)',
    borderRadius: 12, padding: 24, width: 950, maxWidth: '96vw',
    border: '1px solid rgba(255,255,255,0.08)', boxShadow: '0 16px 48px rgba(0,0,0,0.4)',
    fontFamily: "'Inter', sans-serif",
    display: 'flex', flexDirection: 'column',
    maxHeight: '90vh'
  };
  const selS: React.CSSProperties = {
    width: '100%', height: 34, borderRadius: 6, border: '1px solid rgba(255,255,255,0.12)',
    background: 'var(--c-surface-container-high, #2a2a3e)', color: 'var(--c-on-surface, #e0e0e0)',
    padding: '0 10px', fontSize: 12, fontFamily: "'Inter', sans-serif", outline: 'none',
  };
  const btnS = (primary?: boolean): React.CSSProperties => ({
    padding: '8px 20px', borderRadius: 6, border: 'none', cursor: 'pointer',
    fontSize: 12, fontWeight: 600, fontFamily: "'Inter', sans-serif",
    background: primary ? 'var(--c-primary, #cfbcff)' : 'rgba(255,255,255,0.06)',
    color: primary ? 'var(--c-on-primary, #1e1e2e)' : 'var(--c-on-surface, #e0e0e0)',
  });



  return (
    <aside className="controls" id="controls-panel">
      {/* Hidden file input */}
      <input type="file" ref={fileInputRef} style={{ display: 'none' }}
        accept=".csv,.json,.jsonl,.tsv" onChange={handleFileSelect}
        onClick={(e) => { (e.target as any).value = null; }} />
      <input type="file" ref={testFileRef} style={{ display: 'none' }}
        accept=".csv,.json,.jsonl,.tsv" onChange={handleTestFile}
        onClick={(e) => { (e.target as any).value = null; }} />

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
            <ParamControl key={p.key} param={p} value={params[p.key]} onChange={(v) => setParam(p.key, v)} />
          ))}

          {basicToggles.length > 0 && (
            <div className="controls__toggles">
              {basicToggles.map((p) => (
                <ParamControl key={p.key} param={p} value={params[p.key]} onChange={(v) => setParam(p.key, v)} />
              ))}
            </div>
          )}
        </section>

        {/* Advanced Section */}
        {mode === 'advanced' && advancedNonToggle.length + advancedToggles.length > 0 && (
          <section className="controls__section controls__section--advanced">
            <header className="controls__section-header">
              <span className="controls__section-title">Advanced</span>
              <Icon name="science" size={12} className="controls__section-icon" />
            </header>
            {advancedNonToggle.map((p) => (
              <ParamControl key={p.key} param={p} value={params[p.key]} onChange={(v) => setParam(p.key, v)} />
            ))}
            {advancedToggles.length > 0 && (
              <div className="controls__toggles">
                {advancedToggles.map((p) => (
                  <ParamControl key={p.key} param={p} value={params[p.key]} onChange={(v) => setParam(p.key, v)} />
                ))}
              </div>
            )}
          </section>
        )}

        {/* Model State Save/Load Section */}
        <section className="controls__section" style={{ borderBottom: '1px solid var(--c-panel-border)', paddingBottom: '12px' }}>
          <header className="controls__section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '3px 8px',
                  fontSize: '10px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  background: 'var(--c-primary-container, rgba(168, 85, 247, 0.15))',
                  border: '1px solid var(--c-primary, #cfbcff)',
                  color: 'var(--c-primary, #cfbcff)',
                  transition: 'all 0.2s ease',
                }}
                onClick={saveModel}
                title="Save trained model weights and data to a file"
              >
                <Icon name="download" size={10} />
                Save .om
              </button>
              <button
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '3px 8px',
                  fontSize: '10px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid var(--c-panel-border, rgba(255,255,255,0.12))',
                  color: 'var(--c-on-surface-variant, #cbc4d2)',
                  transition: 'all 0.2s ease',
                }}
                onClick={() => omFileInputRef.current?.click()}
                title="Open model from a saved .om file"
              >
                <Icon name="upload" size={10} />
                Open .om
              </button>
            </div>
          </header>
          <input
            type="file"
            ref={omFileInputRef}
            style={{ display: 'none' }}
            accept=".om"
            onChange={handleOMUpload}
          />
        </section>

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

          {/* Re-import button when import is active */}
          {datasetId === 'import' && (
            <button
              style={{ ...btnS(), width: '100%', marginTop: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, height: 30 }}
              onClick={() => fileInputRef.current?.click()}
            >
              <Icon name="file_open" size={13} />
              <span style={{ fontSize: 11 }}>Choose File</span>
            </button>
          )}

          {datasetId !== 'import' && model.dataset.params.map((p) => (
            <ParamControl key={p.key} param={p} value={datasetParams[p.key]} onChange={(v) => setDatasetParam(p.key, v)} />
          ))}

          {/* Test Dataset Section */}
          {model.trainable && (
            <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {/* Button A: Evaluate custom uploaded test file */}
              <button
                style={{
                  ...btnS(),
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                  height: 30,
                  background: 'rgba(168, 85, 247, 0.12)',
                  color: '#d8b4fe',
                  border: '0.5px solid rgba(168, 85, 247, 0.25)',
                  cursor: 'pointer'
                }}
                onClick={() => {
                  if (datasetId !== 'import' || !importValidation?.valid) {
                    setTestError('Please configure your main training dataset import first so we know how to parse the test data.');
                    return;
                  }
                  testFileRef.current?.click();
                }}
              >
                <Icon name="upload_file" size={13} />
                <span style={{ fontSize: 11, fontWeight: 600 }}>Evaluate Test File</span>
              </button>

              {/* Button B: Evaluate on already split test data */}
              {testData && testData.length > 0 && (
                <button
                  style={{
                    ...btnS(),
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 6,
                    height: 30,
                    background: 'rgba(16, 185, 129, 0.12)',
                    color: '#a7f3d0',
                    border: '0.5px solid rgba(16, 185, 129, 0.25)',
                    cursor: 'pointer'
                  }}
                  onClick={() => {
                    setShowTestResults(true);
                  }}
                >
                  <Icon name="insights" size={13} />
                  <span style={{ fontSize: 11, fontWeight: 600 }}>Evaluate Test Split ({testData.length} pts)</span>
                </button>
              )}

              {testError && (
                <div style={{ color: '#ff8a80', fontSize: 11, background: 'rgba(255,138,128,0.08)', padding: '6px 12px', borderRadius: 4 }}>
                  {testError}
                </div>
              )}
            </div>
          )}
        </section>
      </div>

      {/* Footer — Train/Reset */}
      {model.trainable && (
        <div className="controls__footer">
          <div className="controls__train-actions">
            <button
              className={`controls__train-btn ${isTraining ? 'controls__train-btn--stop' : ''}`}
              onClick={isTraining ? stopTraining : startTraining}
              id="btn-train"
            >
              <Icon name={isTraining ? 'stop' : 'play_arrow'} size={14} />
              <span>{isTraining ? 'Stop' : 'Train'}</span>
            </button>
            <button className="controls__reset-btn" onClick={() => setShowResetConfirm(true)} disabled={isTraining} id="btn-reset">
              <Icon name="refresh" size={14} />
            </button>
          </div>
        </div>
      )}

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

      {/* Import Error Toast */}
      {importError && !showImportPopup && (
        <div style={{
          position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', zIndex: 10000,
          background: '#ff5252', color: '#fff', padding: '10px 20px', borderRadius: 8,
          fontSize: 12, fontFamily: "'Inter', sans-serif", boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
        }}>
          {importError}
        </div>
      )}

      {/* Import Mapping Popup */}
      {showImportPopup && importResult && (
        <div style={overlayS} onClick={() => setShowImportPopup(false)}>
          <div style={dialogS} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <Icon name="dataset" size={22} style={{ color: 'var(--c-primary)' }} />
              <h3 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>Import Dataset</h3>
            </div>
            <p style={{ fontSize: 11, opacity: 0.5, margin: '0 0 16px', lineHeight: 1.5 }}>
              {importResult.fileName} · {importValidation?.info}
            </p>

            {/* Validation badge */}
            {importValidation && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', borderRadius: 6, marginBottom: 16,
                background: importValidation.valid ? 'rgba(76,175,80,0.1)' : 'rgba(255,138,128,0.1)',
                border: `1px solid ${importValidation.valid ? 'rgba(76,175,80,0.25)' : 'rgba(255,138,128,0.25)'}`,
              }}>
                <Icon name={importValidation.valid ? 'check_circle' : 'warning'} size={15}
                  style={{ color: importValidation.valid ? '#4caf50' : '#ff8a80' }} />
                <span style={{ fontSize: 11, color: importValidation.valid ? '#4caf50' : '#ff8a80' }}>
                  {importValidation.valid ? `Compatible with ${model?.name}` : importValidation.error}
                </span>
              </div>
            )}

            <div style={{ display: 'flex', gap: 24, alignItems: 'stretch', flex: 1, minHeight: 0 }}>
              {/* Left Column - Configuration & Help */}
              <div style={{ flex: '0 0 350px', display: 'flex', flexDirection: 'column', gap: 14, overflowY: 'auto', paddingRight: '4px' }}>

                {/* Educational model help card */}
                <div style={{
                  background: 'rgba(207, 188, 255, 0.04)',
                  border: '1px solid rgba(207, 188, 255, 0.1)',
                  borderRadius: 8,
                  padding: 12,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 6
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--c-primary)', fontWeight: 600, fontSize: 12 }}>
                    <Icon name="science" size={16} />
                    <span>{modelConfig.title}</span>
                  </div>
                  <span style={{ fontSize: 11, opacity: 0.8, lineHeight: '1.4em' }}>
                    {modelConfig.helpText}
                  </span>
                  <span style={{ fontSize: 10, opacity: 0.5, lineHeight: '1.3em', fontStyle: 'italic', marginTop: 2 }}>
                    {modelConfig.details}
                  </span>
                </div>

                {/* Categorical Encoding Section */}
                {isMultiInput && importResult.categoricalColumns.length > 0 && (
                  <>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#e7c365', marginBottom: 2 }}>
                      Categorical Columns — Encoding
                    </div>
                    <div style={{ fontSize: 10, opacity: 0.5, marginTop: -8, marginBottom: 4 }}>
                      Encode categorical columns to use them as numeric features
                    </div>
                    {importResult.categoricalColumns
                      .filter(col => col !== importMapping.label)
                      .map(col => {
                        const uniqueVals = [...new Set(importResult.data.slice(0, 50).map(r => String(r[col] ?? '')))];
                        return (
                          <div key={col} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <div style={{ fontSize: 11, fontWeight: 600, opacity: 0.85, marginBottom: 2, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {col}
                                <span style={{ fontSize: 9, opacity: 0.5, marginLeft: 6 }}>
                                  ({uniqueVals.length} unique)
                                </span>
                              </div>
                            </div>
                            <select
                              style={{ ...selS, width: 130, flexShrink: 0 }}
                              value={catEncodings[col] || 'skip'}
                              onChange={e => setCatEncodings(prev => ({ ...prev, [col]: e.target.value as any }))}
                            >
                              <option value="skip">Skip</option>
                              <option value="one-hot">One-Hot Encode</option>
                              <option value="label">Label Encode</option>
                            </select>
                          </div>
                        );
                      })}
                    <div style={{ height: 1, background: 'rgba(255,255,255,0.06)', margin: '4px 0' }} />
                  </>
                )}

                {/* Mapping dropdowns */}
                {isMultiInput ? (
                  <>
                    <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--c-primary)', marginBottom: 6 }}>
                      Map Input Features ({numInputs} required)
                    </div>
                    {Array.from({ length: numInputs }).map((_, i) => (
                      <div key={i}>
                        <label style={{ fontSize: 11, fontWeight: 600, opacity: 0.6, marginBottom: 5, display: 'block' }}>
                          Input Node {i} Feature
                        </label>
                        <select
                          style={selS}
                          value={importMapping.features?.[i] || ''}
                          onChange={e => {
                            const val = e.target.value;
                            setImportMapping(prev => {
                              const newF = [...(prev.features || [])];
                              newF[i] = val;
                              return { ...prev, features: newF };
                            });
                          }}
                        >
                          <option value="" disabled>Select a feature...</option>
                          {availableFeatureCols.filter(c => c !== importMapping.label).map(c => (
                            <option key={c} value={c}>{c}</option>
                          ))}
                        </select>
                      </div>
                    ))}
                  </>
                ) : (
                  <>
                    <div>
                      <label style={{ fontSize: 11, fontWeight: 600, opacity: 0.6, marginBottom: 5, display: 'block' }}>
                        {modelConfig.xLabel}
                      </label>
                      <select style={selS} value={importMapping.x}
                        onChange={e => setImportMapping(prev => ({ ...prev, x: e.target.value }))}>
                        {importResult.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </div>
                    <div>
                      <label style={{ fontSize: 11, fontWeight: 600, opacity: 0.6, marginBottom: 5, display: 'block' }}>
                        {modelConfig.yLabel}
                      </label>
                      <select style={selS} value={importMapping.y}
                        onChange={e => setImportMapping(prev => ({ ...prev, y: e.target.value }))}>
                        {importResult.numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </div>
                  </>
                )}

                {/* Dynamic Label / Target Column */}
                {modelConfig.showTarget && (
                  <div>
                    <label style={{ fontSize: 11, fontWeight: 600, opacity: 0.6, marginBottom: 5, display: 'block' }}>
                      {modelConfig.targetLabel}
                    </label>
                    <select style={selS} value={importMapping.label || ''}
                      onChange={e => setImportMapping(prev => ({ ...prev, label: e.target.value }))}>
                      {importResult.columns.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                )}

                {/* Auto-Scale Toggle Card */}
                <div style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.04)',
                  borderRadius: 8,
                  padding: 10,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  marginTop: 10
                }}>
                  <input
                    type="checkbox"
                    id="import-autoscale"
                    checked={autoScale}
                    onChange={e => setAutoScale(e.target.checked)}
                    style={{ cursor: 'pointer', accentColor: 'var(--c-primary)' }}
                  />
                  <label htmlFor="import-autoscale" style={{ fontSize: 11, fontWeight: 500, opacity: 0.8, cursor: 'pointer', userSelect: 'none', lineHeight: '1.3em' }}>
                    Auto-scale features to [0, 1] range (recommended)
                  </label>
                </div>

                {/* Train / Test Split Card */}
                <div style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.04)',
                  borderRadius: 8,
                  padding: 10,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8,
                  marginTop: 10
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input
                      type="checkbox"
                      id="import-autosplit"
                      checked={autoSplit}
                      onChange={e => setAutoSplit(e.target.checked)}
                      style={{ cursor: 'pointer', accentColor: 'var(--c-primary)' }}
                    />
                    <label htmlFor="import-autosplit" style={{ fontSize: 11, fontWeight: 500, opacity: 0.8, cursor: 'pointer', userSelect: 'none', lineHeight: '1.3em' }}>
                      Auto-split to Train and Test sets
                    </label>
                  </div>

                  {autoSplit && (
                    <div style={{ paddingLeft: 20 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, opacity: 0.7, marginBottom: 4 }}>
                        <span>Train Ratio: {Math.round(trainRatio * 100)}%</span>
                        <span>Test Ratio: {Math.round((1 - trainRatio) * 100)}%</span>
                      </div>
                      <input
                        type="range"
                        min={0.5}
                        max={0.9}
                        step={0.05}
                        value={trainRatio}
                        onChange={e => setTrainRatio(parseFloat(e.target.value))}
                        style={{ width: '100%', accentColor: 'var(--c-primary)', cursor: 'pointer' }}
                      />
                    </div>
                  )}
                </div>
              </div>

              {/* Right Column - Data Preview & Visual Scatter Plot */}
              <div style={{ flex: '1 1 auto', background: 'var(--c-surface-container-high)', borderRadius: 8, border: '1px solid var(--c-panel-border)', overflow: 'hidden', display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                <div style={{ padding: '10px 14px', background: 'rgba(255,255,255,0.02)', fontSize: 12, fontWeight: 600, borderBottom: '1px solid var(--c-panel-border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span>Data Preview</span>
                    <div style={{ display: 'flex', alignItems: 'center', background: 'rgba(0,0,0,0.2)', borderRadius: 20, padding: 2, border: '1px solid rgba(255,255,255,0.05)' }}>
                      <button
                        onClick={() => setPreviewTab('plot')}
                        style={{ padding: '4px 10px', fontSize: 10, borderRadius: 18, border: 'none', background: previewTab === 'plot' ? 'var(--c-surface-container-highest)' : 'transparent', color: previewTab === 'plot' ? 'var(--c-on-surface)' : 'var(--c-on-surface-variant)', cursor: 'pointer', transition: 'all 0.2s', fontWeight: previewTab === 'plot' ? 600 : 400 }}
                      >
                        📊 Visual Plot
                      </button>
                      <button
                        onClick={() => setPreviewTab('raw')}
                        style={{ padding: '4px 10px', fontSize: 10, borderRadius: 18, border: 'none', background: previewTab === 'raw' ? 'var(--c-surface-container-highest)' : 'transparent', color: previewTab === 'raw' ? 'var(--c-on-surface)' : 'var(--c-on-surface-variant)', cursor: 'pointer', transition: 'all 0.2s', fontWeight: previewTab === 'raw' ? 600 : 400 }}
                      >
                        📋 Raw Table
                      </button>
                      <button
                        onClick={() => setPreviewTab('vector')}
                        style={{ padding: '4px 10px', fontSize: 10, borderRadius: 18, border: 'none', background: previewTab === 'vector' ? 'var(--c-surface-container-highest)' : 'transparent', color: previewTab === 'vector' ? 'var(--c-on-surface)' : 'var(--c-on-surface-variant)', cursor: 'pointer', transition: 'all 0.2s', fontWeight: previewTab === 'vector' ? 600 : 400 }}
                      >
                        🔢 Input Vector
                      </button>
                    </div>
                  </div>
                  <span style={{ opacity: 0.5, fontWeight: 400, fontSize: 11 }}>
                    {previewTab === 'plot' ? (previewData ? `Plotting ${previewData.length} points` : 'Configure features...') : previewTab === 'vector' ? (previewData ? `Showing first ${Math.min(15, previewData.length)} scaled rows` : 'Configure features...') : `Showing first 15 rows`}
                  </span>
                </div>

                <div style={{ flex: 1, overflow: 'auto' }}>
                  {previewTab === 'plot' ? (
                    <ImportScatterPlot
                      data={previewData || []}
                      modelType={PlaygroundDataEngine.getModelDataType(activeModelId, params)}
                      isMultiInput={isMultiInput}
                      importMapping={importMapping}
                    />
                  ) : previewTab === 'vector' ? (
                    !previewData ? (
                      <div style={{ padding: 24, textAlign: 'center', opacity: 0.5, fontSize: 11 }}>
                        Finish configuring features to see the final scaled input values.
                      </div>
                    ) : (
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, whiteSpace: 'nowrap' }}>
                        <thead style={{ position: 'sticky', top: 0, background: 'var(--c-surface-container-high)', zIndex: 1, boxShadow: '0 1px 2px rgba(0,0,0,0.2)' }}>
                          <tr>
                            <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-on-surface-variant)', width: 40 }}>#</th>
                            {isMultiInput ? (
                              <>
                                {Array.from({ length: numInputs }).map((_, j) => (
                                  <th key={j} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-primary)' }}>
                                    F{j} <span style={{ opacity: 0.5, fontWeight: 400 }}>({importMapping.features?.[j]})</span>
                                  </th>
                                ))}
                                {(PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'classification' || isMultiInput) && (
                                  <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-tertiary)' }}>
                                    {PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'regression' ? 'Target' : 'Label'} <span style={{ opacity: 0.5, fontWeight: 400 }}>({importMapping.label})</span>
                                  </th>
                                )}
                              </>
                            ) : (
                              <>
                                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-primary)' }}>X</th>
                                <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-primary)' }}>Y</th>
                                {PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'classification' && <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-tertiary)' }}>Class</th>}
                                {PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'regression' && <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-tertiary)' }}>Value</th>}
                              </>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {previewData.slice(0, 15).map((row, i) => (
                            <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                              <td style={{ padding: '6px 8px', color: 'var(--c-on-surface-variant)', opacity: 0.5 }}>{i + 1}</td>
                              {isMultiInput ? (
                                <>
                                  {row.features.map((v: number, j: number) => (
                                    <td key={j} style={{ padding: '6px 12px', fontFamily: "'JetBrains Mono', monospace", opacity: 0.85, color: 'var(--c-on-surface)' }}>
                                      {v.toFixed(4)}
                                    </td>
                                  ))}
                                  {(PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'classification' || isMultiInput) && (
                                    <td style={{ padding: '6px 12px', fontFamily: "'JetBrains Mono', monospace", opacity: 0.85, color: 'var(--c-tertiary)' }}>
                                      {PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'regression' ? row.label?.toFixed(4) : (row.label ?? row.cls ?? 0)}
                                    </td>
                                  )}
                                </>
                              ) : (
                                <>
                                  <td style={{ padding: '6px 12px', fontFamily: "'JetBrains Mono', monospace", opacity: 0.85 }}>{row.x.toFixed(4)}</td>
                                  <td style={{ padding: '6px 12px', fontFamily: "'JetBrains Mono', monospace", opacity: 0.85 }}>{row.y.toFixed(4)}</td>
                                  {PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'classification' && <td style={{ padding: '6px 12px', fontFamily: "'JetBrains Mono', monospace", opacity: 0.85, color: 'var(--c-tertiary)' }}>{row.cls}</td>}
                                  {PlaygroundDataEngine.getModelDataType(activeModelId, params) === 'regression' && <td style={{ padding: '6px 12px', fontFamily: "'JetBrains Mono', monospace", opacity: 0.85, color: 'var(--c-tertiary)' }}>{row.label?.toFixed(4)}</td>}
                                </>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )
                  ) : (
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, whiteSpace: 'nowrap' }}>
                      <thead style={{ position: 'sticky', top: 0, background: 'var(--c-surface-container-high)', zIndex: 1, boxShadow: '0 1px 2px rgba(0,0,0,0.2)' }}>
                        <tr>
                          <th style={{ padding: '8px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: 'var(--c-on-surface-variant)', width: 40 }}>#</th>
                          {importResult.columns.map(c => {
                            const isLabel = importMapping.label === c;
                            const isFeature = (importMapping.features || []).includes(c) || (!isMultiInput && (importMapping.x === c || importMapping.y === c));
                            return (
                              <th key={c} style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '1px solid var(--c-panel-border)', color: isLabel ? 'var(--c-tertiary)' : isFeature ? 'var(--c-primary)' : 'var(--c-on-surface-variant)' }}>
                                {c} {isLabel && <span style={{ opacity: 0.7, fontWeight: 400 }}>(Label)</span>}
                                {isFeature && !isLabel && <span style={{ opacity: 0.7, fontWeight: 400 }}>(Feature)</span>}
                              </th>
                            );
                          })}
                        </tr>
                      </thead>
                      <tbody>
                        {importResult.data.slice(0, 15).map((row, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}>
                            <td style={{ padding: '6px 8px', color: 'var(--c-on-surface-variant)', opacity: 0.5 }}>{i + 1}</td>
                            {importResult.columns.map(c => {
                              const val = row[c];
                              const isNumeric = typeof val === 'number' || !isNaN(Number(val));
                              return (
                                <td key={c} style={{ padding: '6px 12px', fontFamily: isNumeric ? "'JetBrains Mono', monospace" : 'inherit', opacity: 0.85 }}>
                                  {val != null ? String(val) : <span style={{ opacity: 0.4 }}>null</span>}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            </div>

            {/* Error & Actions Footer */}
            <div style={{ marginTop: 20, paddingTop: 16, borderTop: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ flex: 1 }}>
                {importError && (
                  <div style={{ color: '#ff8a80', fontSize: 11, background: 'rgba(255,138,128,0.08)', padding: '6px 12px', borderRadius: 4, display: 'inline-block' }}>
                    {importError}
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <button style={btnS()} onClick={() => setShowImportPopup(false)}>Cancel</button>
                <button
                  style={{ ...btnS(true), opacity: importValidation?.valid ? 1 : 0.5, cursor: importValidation?.valid ? 'pointer' : 'not-allowed' }}
                  onClick={importValidation?.valid ? handleApplyImport : undefined}
                >
                  Apply & Import
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Results Popup */}
      {showTestResults && (
        <div style={overlayS} onClick={(e) => e.target === e.currentTarget && setShowTestResults(false)}>
          <div style={{ ...dialogS, width: 600, maxHeight: '80vh', overflowY: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
              <div>
                <h3 style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>Test Evaluation Results</h3>
                <div style={{ fontSize: 12, color: 'var(--c-on-surface-variant)', marginTop: 4 }}>
                  {testResults ? `Evaluated on ${testResults.total} test samples` : 'Evaluating...'}
                </div>
              </div>
              <button style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer', opacity: 0.7 }} onClick={() => setShowTestResults(false)}>
                <Icon name="close" size={20} />
              </button>
            </div>

            {!testResults ? (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--c-primary)' }}>
                <Icon name="sync" size={32} className="spin" />
                <div style={{ marginTop: 16 }}>Running inference on test dataset...</div>
              </div>
            ) : (
              <>
                {testResults.type === 'regression' && (
                  <>
                    <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
                      <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '24px' }}>{(Math.max(0, testResults.r2) * 100).toFixed(2)}%</div>
                      <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>R² Fit Quality (Accuracy of Fit)</div>
                      <div style={{ fontSize: '10px', color: 'var(--c-on-surface-variant)', opacity: 0.7, textAlign: 'center' }}>Percentage of target variance explained by the model's predictions</div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 20 }}>
                      <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                        <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>R² Score (Decimal)</div>
                        <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--c-primary)' }}>{testResults.r2.toFixed(4)}</div>
                      </div>
                      <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                        <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Mean Squared Error (MSE)</div>
                        <div style={{ fontSize: 24, fontWeight: 600 }}>{testResults.mse.toFixed(4)}</div>
                      </div>
                      <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                        <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Root Mean Squared Error (RMSE)</div>
                        <div style={{ fontSize: 24, fontWeight: 600 }}>{testResults.rmse.toFixed(4)}</div>
                      </div>
                      <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                        <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Mean Absolute Error (MAE)</div>
                        <div style={{ fontSize: 24, fontWeight: 600 }}>{testResults.mae.toFixed(4)}</div>
                      </div>
                    </div>
                    <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px 14px', borderRadius: 8, fontSize: 11, border: '1px solid rgba(255,255,255,0.04)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
                      <span style={{ color: 'var(--c-on-surface-variant)' }}>Secondary Metric: Accuracy (Within ±0.5 Range)</span>
                      <span style={{ fontWeight: 'bold', color: 'var(--c-tertiary)', fontSize: 13 }}>{(testResults.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  </>
                )}

                {testResults.type === 'clustering' && (
                  <>
                    <div style={{ background: 'var(--c-surface-variant)', padding: '12px', borderRadius: '6px', border: '1px solid var(--c-panel-border)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', marginBottom: '16px' }}>
                      <div style={{ color: 'var(--c-primary)', fontWeight: 'bold', fontSize: '24px' }}>{testResults.avgCentroidDist.toFixed(4)}</div>
                      <div style={{ color: 'var(--c-on-surface-variant)', fontWeight: '600', marginTop: '4px' }}>Average Centroid / Core Point Distance</div>
                      <div style={{ fontSize: '10px', color: 'var(--c-on-surface-variant)', opacity: 0.7, textAlign: 'center' }}>Mean Euclidean distance from test points to their assigned cluster reference</div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(1, 1fr)', gap: 12, marginBottom: 20 }}>
                      <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                        <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Silhouette Score or Clustering Quality Indicator</div>
                        <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--c-tertiary)' }}>{testResults.silhouetteScore.toFixed(4)}</div>
                      </div>
                    </div>
                  </>
                )}

                {(testResults.type === 'binary' || testResults.type === 'multiclass') && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, marginBottom: 20 }}>
                    <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Accuracy</div>
                      <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--c-primary)' }}>{(testResults.accuracy * 100).toFixed(2)}%</div>
                    </div>
                    {testResults.type === 'binary' ? (
                      <>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Precision</div>
                          <div style={{ fontSize: 24, fontWeight: 600 }}>{(testResults.precision * 100).toFixed(2)}%</div>
                        </div>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Recall</div>
                          <div style={{ fontSize: 24, fontWeight: 600 }}>{(testResults.recall * 100).toFixed(2)}%</div>
                        </div>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>F1 Score</div>
                          <div style={{ fontSize: 24, fontWeight: 600 }}>{(testResults.f1 * 100).toFixed(2)}%</div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Macro Precision</div>
                          <div style={{ fontSize: 24, fontWeight: 600 }}>{(testResults.macroPrecision * 100).toFixed(2)}%</div>
                        </div>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Macro Recall</div>
                          <div style={{ fontSize: 24, fontWeight: 600 }}>{(testResults.macroRecall * 100).toFixed(2)}%</div>
                        </div>
                        <div style={{ background: 'rgba(255,255,255,0.03)', padding: 12, borderRadius: 8 }}>
                          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Macro F1 Score</div>
                          <div style={{ fontSize: 24, fontWeight: 600 }}>{(testResults.macroF1 * 100).toFixed(2)}%</div>
                        </div>
                      </>
                    )}
                  </div>
                )}

                {testResults.confusionMatrix && (
                  <div style={{ marginTop: 20 }}>
                    <h4 style={{ margin: '0 0 10px 0', fontSize: 13, color: 'var(--c-on-surface-variant)' }}>Confusion Matrix</h4>
                    <div style={{ overflowX: 'auto', background: 'rgba(0,0,0,0.2)', padding: 12, borderRadius: 8 }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', fontSize: 12 }}>
                        <thead>
                          <tr>
                            <th style={{ padding: 8, color: 'var(--c-on-surface-variant)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>True \ Pred</th>
                            {testResults.confusionMatrix.map((_: any, i: number) => (
                              <th key={i} style={{ padding: 8, color: 'var(--c-primary)', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>Class {i}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {testResults.confusionMatrix.map((row: number[], i: number) => (
                            <tr key={i}>
                              <th style={{ padding: 8, color: 'var(--c-tertiary)', borderRight: '1px solid rgba(255,255,255,0.1)' }}>Class {i}</th>
                              {row.map((val: number, j: number) => (
                                <td key={j} style={{ padding: 8, background: val > 0 ? `rgba(168, 85, 247, ${Math.min(val / testResults.total * 2, 0.8)})` : 'transparent' }}>
                                  {val}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}
