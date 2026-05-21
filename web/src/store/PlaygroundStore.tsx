/**
 * Playground state management.
 * Manages: active model, params, dataset, code tab, training, metrics, mode, code panel.
 */

import { createContext, useContext, useState, useCallback, useEffect, useRef, type ReactNode } from 'react';
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
  /** Incremented on full reset — visualizations watch this to clear state */
  resetVersion: number;
  /** Imported data points from file */
  importedData: any[] | null;
  /** Incremented when import data changes */
  importVersion: number;
  /** Stores scaling statistics for imported data */
  importStats: { mins: number[]; maxs: number[] } | null;
  /** Test dataset points for evaluation */
  testData: any[] | null;
  /** Test evaluation results from visualization */
  testResults: Record<string, any> | null;
  /** Incremented when test data changes */
  testVersion: number;
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
  setImportedData: (data: any[] | null) => void;
  setImportStats: (stats: { mins: number[]; maxs: number[]; targetMin?: number; targetMax?: number } | null) => void;
  setTestData: (data: any[] | null) => void;
  setTestResults: (results: Record<string, any> | null) => void;
}

export const PlaygroundContext = (() => {
  if (!(globalThis as any).__PLAYGROUND_CONTEXT__) {
    (globalThis as any).__PLAYGROUND_CONTEXT__ = createContext<PlaygroundContextValue | null>(null);
  }
  return (globalThis as any).__PLAYGROUND_CONTEXT__ as React.Context<PlaygroundContextValue | null>;
})();

function buildDefaults(params: ParamDescriptor[]): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  for (const p of params) {
    defaults[p.key] = p.defaultValue;
  }
  return defaults;
}

function readURLParams() {
  const query = new URLSearchParams(window.location.search);
  const parsedParams: Record<string, unknown> = {};
  const parsedDatasetParams: Record<string, unknown> = {};
  
  if (query.has('params')) {
    try { Object.assign(parsedParams, JSON.parse(query.get('params')!)); } catch (e) {}
  }
  if (query.has('datasetParams')) {
    try { Object.assign(parsedDatasetParams, JSON.parse(query.get('datasetParams')!)); } catch (e) {}
  }
  
  return {
    model: query.get('model'),
    dataset: query.get('dataset'),
    params: parsedParams,
    datasetParams: parsedDatasetParams,
  };
}

export function PlaygroundProvider({ children, initialModelId }: { children: ReactNode; initialModelId: string }) {
  const initialURL = readURLParams();
  const [activeModelId, setActiveModelIdRaw] = useState(initialURL.model || initialModelId);
  const [model, setModel] = useState<ModelDescriptor | null>(null);
  const [params, setParams] = useState<Record<string, unknown>>(initialURL.params);
  const [datasetId, setDatasetId] = useState(initialURL.dataset || '');
  const [datasetParams, setDatasetParams] = useState<Record<string, unknown>>(initialURL.datasetParams);
  const [activeCodeTab, setActiveCodeTab] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<MetricValue[]>([]);
  const [mode, setModeRaw] = useState<ParamLevel>(() => {
    const stored = localStorage.getItem('omodels-mode');
    return (stored === 'advanced' ? 'advanced' : 'basic') as ParamLevel;
  });
  const [isCodePanelOpen, setIsCodePanelOpen] = useState(true);
  const [resetVersion, setResetVersion] = useState(0);
  const [importedData, setImportedDataRaw] = useState<any[] | null>(null);
  const [importVersion, setImportVersion] = useState(0);
  const [importStats, setImportStats] = useState<{ mins: number[]; maxs: number[]; targetMin?: number; targetMax?: number } | null>(null);
  const [testData, setTestDataRaw] = useState<any[] | null>(null);
  const [testResults, setTestResults] = useState<Record<string, any> | null>(null);
  const [testVersion, setTestVersion] = useState(0);

  const isFirstLoad = useRef(true);

  /* Load model when activeModelId changes */
  useEffect(() => {
    const m = getModel(activeModelId);
    if (!m) return;
    setModel(m);
    
    // Only use URL params on the very first load of the initial model.
    // If switching models manually, always use defaults.
    if (isFirstLoad.current) {
      setParams(prev => Object.keys(prev).length > 0 ? prev : buildDefaults(m.params));
      setDatasetId(prev => prev || m.dataset.defaultDataset);
      setDatasetParams(prev => Object.keys(prev).length > 0 ? prev : buildDefaults(m.dataset.params));
      isFirstLoad.current = false;
    } else {
      setParams(buildDefaults(m.params));
      setDatasetId(m.dataset.defaultDataset);
      setDatasetParams(buildDefaults(m.dataset.params));
    }
    
    setActiveCodeTab(0);
    setIsTraining(false);
    setLiveMetrics(m.defaultMetrics);
  }, [activeModelId]);

  /* Sync state to URL */
  useEffect(() => {
    const t = setTimeout(() => {
      const query = new URLSearchParams(window.location.search);
      query.set('model', activeModelId);
      if (datasetId) query.set('dataset', datasetId);
      if (Object.keys(params).length > 0) query.set('params', JSON.stringify(params));
      if (Object.keys(datasetParams).length > 0) query.set('datasetParams', JSON.stringify(datasetParams));
      
      const newUrl = `${window.location.pathname}?${query.toString()}`;
      window.history.replaceState(null, '', newUrl);
    }, 500);
    return () => clearTimeout(t);
  }, [activeModelId, datasetId, params, datasetParams]);

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
    if (model) {
      setLiveMetrics(model.defaultMetrics);
      setParams(buildDefaults(model.params));
    }
    setResetVersion(v => v + 1);
  }, [model]);
  const setMode = useCallback((m: ParamLevel) => {
    setModeRaw(m);
    localStorage.setItem('omodels-mode', m);
  }, []);
  const toggleCodePanel = useCallback(() => setIsCodePanelOpen((v) => !v), []);
  const setImportedData = useCallback((data: any[] | null) => {
    setImportedDataRaw(data);
    setImportVersion(v => v + 1);
  }, []);
  const setTestData = useCallback((data: any[] | null) => {
    setTestDataRaw(data);
    setTestResults(null);
    setTestVersion(v => v + 1);
  }, []);

  return (
    <PlaygroundContext.Provider
      value={{
        activeModelId, model, params, datasetId, datasetParams,
        activeCodeTab, isTraining, liveMetrics, mode, isCodePanelOpen, resetVersion,
        importedData, importVersion, importStats,
        testData, testResults, testVersion,
        setActiveModel, setParam, setDataset, setDatasetParam,
        setActiveCodeTab, startTraining, stopTraining, resetTraining,
        setLiveMetrics, setMode, toggleCodePanel, setImportedData, setImportStats,
        setTestData, setTestResults,
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
