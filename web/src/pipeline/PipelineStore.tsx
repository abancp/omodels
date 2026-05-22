/**
 * Pipeline Store — React context for pipeline state management
 */

import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from 'react';
import type { Pipeline, PipelineNode, PipelineEdge, NodeState, PipelineState } from './PipelineTypes';
import { validatePipeline, executePipeline, parseCSV, trainTestSplit, topologicalSort } from './PipelineEngine';
import type { ValidationResult, NodeExecutionResult, DataFrame } from './PipelineTypes';
import { NODE_CATALOG } from './NodeCatalog';

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
  csvData: { headers: string[]; rows: number[][]; rowCount: number } | null;
  fittedEngines: Record<string, any>;
  featureNames: string[];

  // Node ops
  addNode: (node: PipelineNode) => void;
  removeNode: (id: string) => void;
  updateNode: (id: string, changes: Partial<PipelineNode>) => void;
  updateNodeConfig: (id: string, config: Partial<any>) => void;
  moveNode: (id: string, pos: { x: number; y: number }) => void;
  selectNode: (id: string | null) => void;

  // Edge ops
  addEdge: (edge: PipelineEdge) => void;
  removeEdge: (id: string) => void;

  // Pipeline ops
  renamePipeline: (name: string) => void;
  runPipeline: () => Promise<void>;
  resetPipeline: () => void;
  savePipeline: () => void;
  loadPipeline: (json: string) => void;

  // Data
  loadCSV: (text: string, filename: string) => void;

  // Inference
  testPrediction: (inputValues: number[]) => number | null;
}

const PipelineContext = createContext<PipelineContextValue | null>(null);


export function PipelineProvider({ children }: { children: ReactNode }) {
  const [pipeline, setPipeline] = useState<Pipeline>(emptyPipeline);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [executionLog, setExecutionLog] = useState<string[]>([]);
  const [nodeResults, setNodeResults] = useState<Record<string, NodeExecutionResult>>({});
  const [isRunning, setIsRunning] = useState(false);
  const [csvData, setCsvData] = useState<{ headers: string[]; rows: number[][]; rowCount: number } | null>(null);
  const [fittedEngines, setFittedEngines] = useState<Record<string, any>>({});
  const [featureNames, setFeatureNames] = useState<string[]>([]);
  const csvRef = useRef<{ headers: string[]; rows: number[][]; rowCount: number } | null>(null);
  // Keep fittedEngines in a ref too for synchronous access in testPrediction
  const fittedEnginesRef = useRef<Record<string, any>>({});
  const topOrderRef = useRef<string[]>([]);

  const touch = useCallback(() => {
    setPipeline(p => ({ ...p, updatedAt: new Date().toISOString() }));
  }, []);

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

  const addEdge = useCallback((edge: PipelineEdge) => {
    setPipeline(p => {
      // Prevent duplicate edges
      const dup = p.edges.find(e => e.source === edge.source && e.target === edge.target);
      if (dup) return p;
      return { ...p, edges: [...p.edges, edge], state: 'draft' };
    });
  }, []);

  const removeEdge = useCallback((id: string) => {
    setPipeline(p => ({ ...p, edges: p.edges.filter(e => e.id !== id), state: 'draft' }));
  }, []);

  const renamePipeline = useCallback((name: string) => {
    setPipeline(p => ({ ...p, name }));
  }, []);

  const runPipeline = useCallback(async () => {
    const v = validatePipeline(pipeline);
    setValidation(v);
    if (!v.valid) {
      setPipeline(p => ({ ...p, state: 'error' }));
      return;
    }

    const sourceNode = pipeline.nodes.find(n => n.family === 'datasource');
    if (!sourceNode) return;
    const cfg = sourceNode.config as any;

    const csv = csvRef.current;
    if (!csv || csv.rows.length === 0) {
      setExecutionLog(['⚠ No CSV data loaded. Upload a dataset using the right panel.']);
      return;
    }

    const targetCol = cfg.targetColumn || csv.headers[csv.headers.length - 1];
    const targetIdx = csv.headers.indexOf(targetCol);
    if (targetIdx === -1) {
      setExecutionLog(['⚠ Target column not found. Check Data Source → Target Column.']);
      return;
    }

    // Split rows into train/test (keeping all columns intact)
    const allRows = csv.rows;
    const n = allRows.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor((i + 1) * 0.582) % (i + 1);
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const splitIdx = Math.floor(n * (1 - (cfg.testSplit ?? 0.2)));
    const trainRows = indices.slice(0, splitIdx).map(i => allRows[i]);
    const testRows = indices.slice(splitIdx).map(i => allRows[i]);

    const trainDF: DataFrame = { columns: [...csv.headers], rows: trainRows };
    const testDF: DataFrame = { columns: [...csv.headers], rows: testRows };

    const fNames = csv.headers.filter(h => h !== targetCol);
    const y = allRows.map(r => r[targetIdx]);

    const pipelineWithData: Pipeline = {
      ...pipeline,
      data: { train: trainDF, test: testDF },
      schema: {
        columns: fNames.map(n => ({ name: n, type: 'numeric' as const })),
        shape: [allRows.length, fNames.length],
        target: targetCol,
        taskType: cfg.taskType === 'auto'
          ? (y.every((v: any) => v === 0 || v === 1) ? 'classification' : 'regression')
          : cfg.taskType,
      },
    };

    setIsRunning(true);
    setExecutionLog([]);
    setNodeResults({});
    setFittedEngines({});
    setPipeline(p => ({ ...p, state: 'training' }));
    setPipeline(p => ({
      ...p,
      nodes: p.nodes.map(n => ({ ...n, state: 'idle' as NodeState })),
    }));

    try {
      const { results, fittedEngines: engines, featureNames: fN } = await executePipeline(
        pipelineWithData,
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

      // Store fitted engines for inference
      fittedEnginesRef.current = engines;
      topOrderRef.current = topologicalSort(pipeline.nodes, pipeline.edges) ?? [];
      setFittedEngines(engines);
      setFeatureNames(fN);

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
    fittedEnginesRef.current = {};
    topOrderRef.current = [];
  }, []);

  // ── Inference: apply preprocessors then model ─────────────────────────────
  const testPrediction = useCallback((inputValues: number[]): number | null => {
    const engines = fittedEnginesRef.current;
    const order = topOrderRef.current;
    if (Object.keys(engines).length === 0 || order.length === 0) return null;

    const nodeMap = new Map(pipeline.nodes.map(n => [n.id, n]));
    let X = [inputValues];
    let result: number | null = null;

    for (const nodeId of order) {
      const engine = engines[nodeId];
      if (!engine) continue;
      const node = nodeMap.get(nodeId);
      if (!node) continue;
      if (node.family === 'preprocessor' && engine.transform) {
        X = engine.transform(X);
      } else if (node.family === 'model' && engine.predict) {
        result = engine.predict(X)[0];
      }
    }
    return result;
  }, [pipeline.nodes]);

  const savePipeline = useCallback(() => {
    const serializable = {
      ...pipeline,
      data: undefined, // don't serialize raw data
    };
    const blob = new Blob([JSON.stringify(serializable, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${pipeline.name.replace(/\s+/g, '-').toLowerCase()}.pipeline.om`;
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

  const loadCSV = useCallback((text: string, _filename: string) => {
    const { headers, rows } = parseCSV(text);
    const data = { headers, rows, rowCount: rows.length };
    setCsvData(data);
    csvRef.current = data;
    // Auto-set target column on source node
    setPipeline(p => ({
      ...p,
      nodes: p.nodes.map(n => n.family === 'datasource' ? {
        ...n,
        config: {
          ...n.config,
          targetColumn: (n.config as any).targetColumn || headers[headers.length - 1] || '',
          datasetName: _filename,
        },
      } : n),
    }));
  }, []);

  return (
    <PipelineContext.Provider value={{
      pipeline, selectedNodeId, validation, executionLog, nodeResults, isRunning, csvData,
      fittedEngines, featureNames,
      addNode, removeNode, updateNode, updateNodeConfig, moveNode, selectNode,
      addEdge, removeEdge,
      renamePipeline, runPipeline, resetPipeline, savePipeline, loadPipeline, loadCSV,
      testPrediction,
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
