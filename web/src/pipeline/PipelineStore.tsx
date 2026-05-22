/**
 * Pipeline Store — React context for pipeline state management
 * 
 * Key change: No global CSV data. Each Dataset node owns its CSV data
 * in its config. Data flows only through connections.
 */

import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from 'react';
import type { Pipeline, PipelineNode, PipelineEdge, NodeState } from './PipelineTypes';
import { CONNECTION_RULES } from './PipelineTypes';
import { validatePipeline, executePipeline, topologicalSort, parseCSV, runInference } from './PipelineEngine';
import type { ValidationResult, NodeExecutionResult, DataPacket } from './PipelineTypes';

let nodeIdCounter = 1;
export function newNodeId() { return `node_${nodeIdCounter++}`; }
let edgeIdCounter = 1;
export function newEdgeId() { return `edge_${edgeIdCounter++}`; }

function emptyPipeline(): Pipeline {
  return {
    id: `pipeline_${Date.now()}`,
    name: 'Untitled Pipeline',
    nodes: [],
    edges: [],
    state: 'draft',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  };
}

interface PipelineContextValue {
  pipeline: Pipeline;
  selectedNodeId: string | null;
  validation: ValidationResult | null;
  executionLog: string[];
  nodeResults: Record<string, NodeExecutionResult>;
  isRunning: boolean;
  fittedEngines: Record<string, any>;
  featureNames: string[];
  rawFeatureNames: string[];

  addNode: (node: PipelineNode) => void;
  removeNode: (id: string) => void;
  updateNode: (id: string, changes: Partial<PipelineNode>) => void;
  updateNodeConfig: (id: string, config: Partial<any>) => void;
  moveNode: (id: string, pos: { x: number; y: number }) => void;
  selectNode: (id: string | null) => void;
  addEdge: (edge: PipelineEdge) => void;
  removeEdge: (id: string) => void;
  renamePipeline: (name: string) => void;
  runPipeline: () => Promise<void>;
  resetPipeline: () => void;
  savePipeline: () => void;
  loadPipeline: (json: string) => void;
  /** Load CSV into a specific Dataset node */
  loadCSVToNode: (nodeId: string, text: string, filename: string) => void;
  /** Check if a proposed connection is valid */
  canConnect: (sourceId: string, targetId: string) => boolean;
  testPrediction: (inputValues: Record<string, number>) => number | null;
}

const PipelineContext = createContext<PipelineContextValue | null>(null);

export function PipelineProvider({ children }: { children: ReactNode }) {
  const [pipeline, setPipeline] = useState<Pipeline>(emptyPipeline);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [executionLog, setExecutionLog] = useState<string[]>([]);
  const [nodeResults, setNodeResults] = useState<Record<string, NodeExecutionResult>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [fittedEngines, setFittedEngines] = useState<Record<string, any>>({});
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const [rawFeatureNames, setRawFeatureNames] = useState<string[]>([]);

  const fittedEnginesRef = useRef<Record<string, any>>({});
  const topOrderRef = useRef<string[]>([]);
  const nodePacketsRef = useRef<Record<string, DataPacket>>({});

  const addNode = useCallback((node: PipelineNode) => {
    setPipeline(p => ({ ...p, nodes: [...p.nodes, node], state: 'draft' }));
  }, []);

  const removeNode = useCallback((id: string) => {
    setPipeline(p => ({
      ...p,
      nodes: p.nodes.filter(n => n.id !== id),
      edges: p.edges.filter(e => e.source !== id && e.target !== id),
      state: 'draft',
    }));
    setSelectedNodeId(s => s === id ? null : s);
  }, []);

  const updateNode = useCallback((id: string, changes: Partial<PipelineNode>) => {
    setPipeline(p => ({
      ...p,
      nodes: p.nodes.map(n => n.id === id ? { ...n, ...changes } : n),
      state: p.state === 'trained' ? 'stale' : p.state,
    }));
  }, []);

  const updateNodeConfig = useCallback((id: string, config: Partial<any>) => {
    setPipeline(p => ({
      ...p,
      nodes: p.nodes.map(n => n.id === id ? { ...n, config: { ...n.config, ...config } } : n),
      state: p.state === 'trained' ? 'stale' : p.state,
    }));
  }, []);

  const moveNode = useCallback((id: string, pos: { x: number; y: number }) => {
    setPipeline(p => ({
      ...p,
      nodes: p.nodes.map(n => n.id === id ? { ...n, position: pos } : n),
    }));
  }, []);

  const selectNode = useCallback((id: string | null) => setSelectedNodeId(id), []);

  /** Check connection validity using CONNECTION_RULES */
  const canConnect = useCallback((sourceId: string, targetId: string): boolean => {
    const srcNode = pipeline.nodes.find(n => n.id === sourceId);
    const tgtNode = pipeline.nodes.find(n => n.id === targetId);
    if (!srcNode || !tgtNode) return false;
    if (sourceId === targetId) return false;
    // Check connection rules
    const allowed = CONNECTION_RULES[tgtNode.family];
    if (!allowed?.includes(srcNode.family)) return false;
    // Check for duplicate edges
    if (pipeline.edges.some(e => e.source === sourceId && e.target === targetId)) return false;
    return true;
  }, [pipeline.nodes, pipeline.edges]);

  const addEdge = useCallback((edge: PipelineEdge) => {
    setPipeline(p => {
      const srcNode = p.nodes.find(n => n.id === edge.source);
      const tgtNode = p.nodes.find(n => n.id === edge.target);
      if (!srcNode || !tgtNode) return p;
      // Enforce connection rules
      const allowed = CONNECTION_RULES[tgtNode.family];
      if (!allowed?.includes(srcNode.family)) return p;
      // Prevent duplicates
      if (p.edges.some(e => e.source === edge.source && e.target === edge.target)) return p;
      return { ...p, edges: [...p.edges, edge], state: 'draft' };
    });
  }, []);

  const removeEdge = useCallback((id: string) => {
    setPipeline(p => ({ ...p, edges: p.edges.filter(e => e.id !== id), state: 'draft' }));
  }, []);

  const renamePipeline = useCallback((name: string) => {
    setPipeline(p => ({ ...p, name }));
  }, []);

  /** Load CSV data into a specific Dataset node's config */
  const loadCSVToNode = useCallback((nodeId: string, text: string, filename: string) => {
    const { headers, rows } = parseCSV(text);
    setPipeline(p => ({
      ...p,
      nodes: p.nodes.map(n => n.id === nodeId ? {
        ...n,
        config: {
          ...n.config,
          csvText: text,
          csvFilename: filename,
          csvHeaders: headers,
          csvRowCount: rows.length,
          targetColumn: (n.config as any).targetColumn || headers[headers.length - 1] || '',
        },
      } : n),
    }));
  }, []);

  const runPipeline = useCallback(async () => {
    const v = validatePipeline(pipeline);
    setValidation(v);
    if (!v.valid) {
      setPipeline(p => ({ ...p, state: 'error' }));
      return;
    }

    setIsRunning(true);
    setExecutionLog([]);
    setNodeResults({});
    setFittedEngines({});
    setPipeline(p => ({
      ...p,
      state: 'training',
      nodes: p.nodes.map(n => ({ ...n, state: 'idle' as NodeState })),
    }));

    try {
      const { results, fittedEngines: engines, featureNames: fN, nodePackets } = await executePipeline(
        pipeline,
        (id) => {
          setPipeline(p => ({
            ...p,
            nodes: p.nodes.map(n => n.id === id ? { ...n, state: 'training' as NodeState } : n),
          }));
        },
        (id, result) => {
          setNodeResults(prev => ({ ...prev, [id]: result }));
          setPipeline(p => ({
            ...p,
            nodes: p.nodes.map(n => n.id === id ? {
              ...n,
              state: result.success ? 'trained' as NodeState : 'error' as NodeState,
              fittedParams: result.fittedParams,
              metrics: result.metrics,
              error: result.error,
            } : n),
          }));
        },
        (msg) => setExecutionLog(prev => [...prev, msg]),
      );

      fittedEnginesRef.current = engines;
      topOrderRef.current = topologicalSort(pipeline.nodes, pipeline.edges) ?? [];
      nodePacketsRef.current = nodePackets;
      setFittedEngines(engines);
      setFeatureNames(fN);
      
      const rawDatasetPacket = Object.values(nodePackets).find((p: any) => p.meta?.targetColumn !== undefined);
      const rawFN = rawDatasetPacket ? rawDatasetPacket.train.columns.filter((c: string) => c !== rawDatasetPacket.meta.targetColumn) : [];
      setRawFeatureNames(rawFN);

      const allMetrics: Record<string, number> = {};
      Object.values(results).forEach(r => {
        if (r.metrics) Object.assign(allMetrics, r.metrics);
      });

      setPipeline(p => ({ ...p, state: 'trained', metrics: allMetrics }));
      setExecutionLog(prev => [...prev, '✓ Pipeline training complete!']);
    } catch (e: any) {
      setPipeline(p => ({ ...p, state: 'error' }));
      setExecutionLog(prev => [...prev, `✗ Error: ${e.message}`]);
    } finally {
      setIsRunning(false);
    }
  }, [pipeline]);

  const resetPipeline = useCallback(() => {
    setPipeline(p => ({
      ...p,
      state: 'draft',
      nodes: p.nodes.map(n => ({ ...n, state: 'idle', fittedParams: undefined, metrics: undefined, error: undefined })),
      metrics: undefined,
    }));
    setNodeResults({});
    setExecutionLog([]);
    setValidation(null);
    setFittedEngines({});
    setFeatureNames([]);
    setRawFeatureNames([]);
    fittedEnginesRef.current = {};
    topOrderRef.current = [];
    nodePacketsRef.current = {};
  }, []);

  const testPrediction = useCallback((inputValues: Record<string, number>): number | null => {
    const engines = fittedEnginesRef.current;
    if (Object.keys(engines).length === 0) return null;
    return runInference(pipeline, inputValues, engines, nodePacketsRef.current);
  }, [pipeline]);

  const savePipeline = useCallback(() => {
    // Strip csvText from config before saving to reduce file size
    const serializable = {
      ...pipeline,
      nodes: pipeline.nodes.map(n => ({
        ...n,
        config: n.family === 'dataset' ? { ...n.config, csvText: undefined } : n.config,
      })),
    };
    const blob = new Blob([JSON.stringify(serializable, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${pipeline.name.replace(/\s+/g, '-').toLowerCase()}.pipeline.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [pipeline]);

  const loadPipeline = useCallback((json: string) => {
    const p = JSON.parse(json) as Pipeline;
    nodeIdCounter = p.nodes.reduce((max, n) => {
      const num = parseInt(n.id.split('_')[1] ?? '0');
      return Math.max(max, num + 1);
    }, nodeIdCounter);
    setPipeline({ ...p, state: 'draft' });
  }, []);

  return (
    <PipelineContext.Provider value={{
      pipeline, selectedNodeId, validation, executionLog, nodeResults, isRunning,
      fittedEngines, featureNames,
      addNode, removeNode, updateNode, updateNodeConfig, moveNode, selectNode,
      addEdge, removeEdge,
      renamePipeline, runPipeline, resetPipeline, savePipeline, loadPipeline,
      loadCSVToNode, canConnect, testPrediction, rawFeatureNames,
    }}>
      {children}
    </PipelineContext.Provider>
  );
}

export function usePipeline(): PipelineContextValue {
  const ctx = useContext(PipelineContext);
  if (!ctx) throw new Error('usePipeline must be used inside PipelineProvider');
  return ctx;
}
