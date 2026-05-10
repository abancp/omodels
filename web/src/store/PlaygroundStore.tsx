/**
 * Playground state management.
 * Manages: active model, params, dataset, code tab, training, metrics, mode, code panel.
 */

import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from 'react';
import { getModel, type ModelDescriptor, type ParamDescriptor, type MetricValue, type ParamLevel } from '../models';

interface PlaygroundState {
  activeModelId: string;
  model: ModelDescriptor | null;
  params: Record<string, unknown>;
  datasetId: string;
  datasetParams: Record<string, unknown>;
  activeCodeTab: number;
  isTraining: boolean;
  liveMetrics: MetricValue[];
  /** Basic or Advanced mode — controls which params are shown */
  mode: ParamLevel;
  /** Whether the code panel is visible */
  isCodePanelOpen: boolean;
}

interface PlaygroundContextValue extends PlaygroundState {
  setActiveModel: (id: string) => void;
  setParam: (key: string, value: unknown) => void;
  setDataset: (id: string) => void;
  setDatasetParam: (key: string, value: unknown) => void;
  setActiveCodeTab: (idx: number) => void;
  startTraining: () => void;
  stopTraining: () => void;
  resetTraining: () => void;
  setLiveMetrics: (metrics: MetricValue[]) => void;
  setMode: (mode: ParamLevel) => void;
  toggleCodePanel: () => void;
}

const PlaygroundContext = createContext<PlaygroundContextValue | null>(null);

function buildDefaults(params: ParamDescriptor[]): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  for (const p of params) {
    defaults[p.key] = p.defaultValue;
  }
  return defaults;
}

export function PlaygroundProvider({ children, initialModelId }: { children: ReactNode; initialModelId: string }) {
  const [activeModelId, setActiveModelIdRaw] = useState(initialModelId);
  const [model, setModel] = useState<ModelDescriptor | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [datasetId, setDatasetId] = useState('');
  const [datasetParams, setDatasetParams] = useState<Record<string, unknown>>({});
  const [activeCodeTab, setActiveCodeTab] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<MetricValue[]>([]);
  const [mode, setModeRaw] = useState<ParamLevel>(() => {
    const stored = localStorage.getItem('omodels-mode');
    return (stored === 'advanced' ? 'advanced' : 'basic') as ParamLevel;
  });
  const [isCodePanelOpen, setIsCodePanelOpen] = useState(true);

  /* Load model when activeModelId changes */
  useEffect(() => {
    const m = getModel(activeModelId);
    if (!m) return;
    setModel(m);
    setParams(buildDefaults(m.params));
    setDatasetId(m.dataset.defaultDataset);
    setDatasetParams(buildDefaults(m.dataset.params));
    setActiveCodeTab(0);
    setIsTraining(false);
    setLiveMetrics(m.defaultMetrics);
  }, [activeModelId]);

  const setActiveModel = useCallback((id: string) => setActiveModelIdRaw(id), []);
  const setParam = useCallback((key: string, value: unknown) => {
    setParams((prev) => ({ ...prev, [key]: value }));
  }, []);
  const setDataset = useCallback((id: string) => setDatasetId(id), []);
  const setDatasetParam = useCallback((key: string, value: unknown) => {
    setDatasetParams((prev) => ({ ...prev, [key]: value }));
  }, []);
  const startTraining = useCallback(() => setIsTraining(true), []);
  const stopTraining = useCallback(() => setIsTraining(false), []);
  const resetTraining = useCallback(() => {
    setIsTraining(false);
    if (model) setLiveMetrics(model.defaultMetrics);
  }, [model]);
  const setMode = useCallback((m: ParamLevel) => {
    setModeRaw(m);
    localStorage.setItem('omodels-mode', m);
  }, []);
  const toggleCodePanel = useCallback(() => setIsCodePanelOpen((v) => !v), []);

  return (
    <PlaygroundContext.Provider
      value={{
        activeModelId, model, params, datasetId, datasetParams,
        activeCodeTab, isTraining, liveMetrics, mode, isCodePanelOpen,
        setActiveModel, setParam, setDataset, setDatasetParam,
        setActiveCodeTab, startTraining, stopTraining, resetTraining,
        setLiveMetrics, setMode, toggleCodePanel,
      }}
    >
      {children}
    </PlaygroundContext.Provider>
  );
}

export function usePlayground(): PlaygroundContextValue {
  const ctx = useContext(PlaygroundContext);
  if (!ctx) throw new Error('usePlayground must be used within PlaygroundProvider');
  return ctx;
}
