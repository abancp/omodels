/**
 * Pipeline Execution Engine
 * 
 * Architecture: Fully isolated per-node data flow.
 * Each node receives data ONLY from its incoming connections via DataPacket.
 * No global dataset — each Dataset node owns its own CSV data.
 */

import type {
  Pipeline, PipelineNode, PipelineEdge, ValidationResult, ValidationError,
  NodeExecutionResult, DataFrame, DataPacket, DataPacketMeta,
} from './PipelineTypes';
import { CONNECTION_RULES } from './PipelineTypes';
import { TFLinearRegressionEngine, TFLogisticRegressionEngine, TFMLPEngine } from './_tf_engines';
import { StandardScalerEngine, MinMaxScalerEngine, SimpleImputerEngine, LabelEncoderEngine, OneHotEncoderEngine, DropColumnsEngine } from './_preprocessors';
import { KNNEngine, DecisionTreeEngine, RandomForestEngine, SVMEngine, NaiveBayesEngine, KMeansEngine } from './_js_models';

// ─── Topological Sort (Kahn's algorithm) ──────────────────────────────────────

export function topologicalSort(nodes: PipelineNode[], edges: PipelineEdge[]): string[] | null {
  const inDeg: Record<string, number> = {};
  const adj: Record<string, string[]> = {};
  nodes.forEach(n => { inDeg[n.id] = 0; adj[n.id] = []; });
  edges.forEach(e => {
    adj[e.source]?.push(e.target);
    inDeg[e.target] = (inDeg[e.target] || 0) + 1;
  });
  const queue = nodes.filter(n => inDeg[n.id] === 0).map(n => n.id);
  const result: string[] = [];
  while (queue.length) {
    const cur = queue.shift()!;
    result.push(cur);
    for (const next of (adj[cur] || [])) {
      inDeg[next]--;
      if (inDeg[next] === 0) queue.push(next);
    }
  }
  return result.length === nodes.length ? result : null;
}

// ─── Validation ────────────────────────────────────────────────────────────────

export function validatePipeline(pipeline: Pipeline): ValidationResult {
  const errors: ValidationError[] = [];
  const { nodes, edges } = pipeline;

  if (!topologicalSort(nodes, edges)) {
    errors.push({ message: 'Pipeline contains a cycle — must be a DAG', type: 'error' });
  }

  // Validate connection rules
  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  edges.forEach(e => {
    const src = nodeMap.get(e.source);
    const tgt = nodeMap.get(e.target);
    if (src && tgt) {
      const allowed = CONNECTION_RULES[tgt.family];
      if (!allowed?.includes(src.family)) {
        errors.push({
          edgeId: e.id,
          message: `Invalid connection: ${src.family} → ${tgt.family}`,
          type: 'error',
        });
      }
    }
  });

  // Dataset nodes must have CSV data
  nodes.filter(n => n.family === 'dataset').forEach(n => {
    const cfg = n.config as any;
    if (!cfg.csvText) {
      errors.push({ nodeId: n.id, message: `"${n.label}" has no CSV data uploaded`, type: 'warning' });
    }
    if (!cfg.targetColumn) {
      errors.push({ nodeId: n.id, message: `"${n.label}" has no target column set`, type: 'warning' });
    }
  });

  // Non-dataset nodes must have incoming connections
  const targets = new Set(edges.map(e => e.target));
  nodes.filter(n => n.family !== 'dataset').forEach(n => {
    if (!targets.has(n.id)) {
      errors.push({ nodeId: n.id, message: `"${n.label}" has no incoming connection`, type: 'error' });
    }
  });

  // Must have at least one dataset
  if (nodes.filter(n => n.family === 'dataset').length === 0) {
    errors.push({ message: 'Pipeline needs at least one Dataset node', type: 'error' });
  }

  return { valid: errors.filter(e => e.type === 'error').length === 0, errors };
}

// ─── CSV Parsing ──────────────────────────────────────────────────────────────

export function parseCSV(text: string): { headers: string[]; rows: any[][] } {
  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };
  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  const rows: any[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(c => c.trim().replace(/^["']|["']$/g, ''));
    rows.push(cols.map(c => { const v = parseFloat(c); return isNaN(v) ? c : v; }));
  }
  return { headers, rows };
}

// ─── Train/Test Split ─────────────────────────────────────────────────────────

function splitData(
  df: DataFrame, testRatio: number, shuffle: boolean
): { train: DataFrame; test: DataFrame } {
  const n = df.rows.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  if (shuffle) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
  }
  const splitIdx = Math.floor(n * (1 - testRatio));
  return {
    train: { columns: [...df.columns], rows: indices.slice(0, splitIdx).map(i => df.rows[i]) },
    test: { columns: [...df.columns], rows: indices.slice(splitIdx).map(i => df.rows[i]) },
  };
}

// ─── Helper: Extract X,y from DataFrame ───────────────────────────────────────

function dfToXY(df: DataFrame, targetCol: string): { X: number[][]; y: number[] } {
  const tIdx = df.columns.indexOf(targetCol);
  if (tIdx === -1) return { X: df.rows.map(r => r.map(Number)), y: [] };
  return {
    X: df.rows.map(row => row.filter((_, i) => i !== tIdx).map(Number)),
    y: df.rows.map(row => Number(row[tIdx])),
  };
}


// ─── Metrics ──────────────────────────────────────────────────────────────────

export function computeClassificationMetrics(yTrue: number[], yPred: number[]) {
  const n = yTrue.length;
  let tp = 0, fp = 0, fn = 0, tn = 0;
  for (let i = 0; i < n; i++) {
    if (yTrue[i] === 1 && yPred[i] === 1) tp++;
    else if (yTrue[i] === 0 && yPred[i] === 1) fp++;
    else if (yTrue[i] === 1 && yPred[i] === 0) fn++;
    else tn++;
  }
  const accuracy = (tp + tn) / (n || 1);
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  return { accuracy, precision, recall, f1 };
}

export function computeRegressionMetrics(yTrue: number[], yPred: number[]) {
  const n = yTrue.length;
  if (!n) return { mse: 0, rmse: 0, mae: 0, r2: 0 };
  const mse = yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0) / n;
  const mae = yTrue.reduce((s, v, i) => s + Math.abs(v - yPred[i]), 0) / n;
  const meanY = yTrue.reduce((a, b) => a + b, 0) / n;
  const ssTot = yTrue.reduce((s, v) => s + (v - meanY) ** 2, 0);
  const ssRes = yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0);
  return { mse, rmse: Math.sqrt(mse), mae, r2: ssTot > 0 ? 1 - ssRes / ssTot : 0 };
}

// ─── Detect task type ─────────────────────────────────────────────────────────

function detectTaskType(y: number[]): 'classification' | 'regression' {
  const unique = new Set(y);
  if (unique.size <= 10 && y.every(v => v === Math.floor(v))) return 'classification';
  return 'regression';
}

// ─── Build DataPacket from Dataset node ───────────────────────────────────────

function buildDatasetPacket(node: PipelineNode): DataPacket {
  const cfg = node.config as any;
  if (!cfg.csvText) throw new Error(`Dataset "${node.label}" has no CSV data`);

  const { headers, rows } = parseCSV(cfg.csvText);
  if (rows.length === 0) throw new Error(`Dataset "${node.label}" has no rows`);

  const targetCol = cfg.targetColumn || headers[headers.length - 1];
  const tIdx = headers.indexOf(targetCol);
  const y = tIdx >= 0 ? rows.map(r => Number(r[tIdx])) : [];
  const taskType = cfg.taskType === 'auto' ? detectTaskType(y) : cfg.taskType;
  const features = headers.filter(h => h !== targetCol);

  const df: DataFrame = { columns: headers, rows };
  // Dataset node outputs unsplit data (train=all, test=empty)
  return {
    train: df,
    test: { columns: headers, rows: [] },
    meta: {
      targetColumn: targetCol,
      taskType,
      featureColumns: features,
      schema: {
        columns: features.map(n => ({ name: n, type: 'numeric' as const })),
        shape: [rows.length, features.length],
        target: targetCol,
        taskType,
      },
    },
  };
}

// ─── Execute Split node ───────────────────────────────────────────────────────

function executeSplit(node: PipelineNode, input: DataPacket): DataPacket {
  const cfg = node.config as any;
  const ratio = cfg.testSplit ?? 0.2;
  const shuffle = cfg.shuffle !== false;

  // Merge train+test from input (dataset outputs all in train)
  const allRows = [...input.train.rows, ...input.test.rows];
  const fullDf: DataFrame = { columns: [...input.train.columns], rows: allRows };
  const { train, test } = splitData(fullDf, ratio, shuffle);

  return { ...input, train, test };
}

// ─── Execute Preprocessor node ────────────────────────────────────────────────

function executePreprocessor(
  node: PipelineNode, input: DataPacket
): { packet: DataPacket; engine: any; fittedParams: Record<string, any> } {
  const cfg = node.config as any;
  const targetCol = input.meta.targetColumn;

  let engine: any = null;
  if (node.type === 'StandardScaler') engine = new StandardScalerEngine();
  else if (node.type === 'MinMaxScaler') engine = new MinMaxScalerEngine();
  else if (node.type === 'SimpleImputer') engine = new SimpleImputerEngine(cfg.params?.strategy);
  else if (node.type === 'LabelEncoder') engine = new LabelEncoderEngine();
  else if (node.type === 'OneHotEncoder') engine = new OneHotEncoderEngine(cfg.params?.dropFirst ?? false);
  else if (node.type === 'DropColumns') engine = new DropColumnsEngine();

  if (!engine || input.train.rows.length === 0) {
    return { packet: input, engine: null, fittedParams: {} };
  }

  // Determine columns to transform
  const selectedCols: string[] = cfg.targetColumns || [];
  let colIndices: number[];
  if (selectedCols.length > 0) {
    colIndices = selectedCols.map((c: string) => input.train.columns.indexOf(c)).filter((i: number) => i !== -1);
  } else if (node.type === 'DropColumns') {
    colIndices = [];
  } else {
    colIndices = input.train.columns.map((_, i) => i).filter(i => input.train.columns[i] !== targetCol);
  }

  if (colIndices.length === 0) return { packet: input, engine: null, fittedParams: {} };

  const extractCols = (rows: any[][]) => rows.map(row => colIndices.map(i => row[i]));
  const subTrain = extractCols(input.train.rows);
  const subTest = input.test.rows.length > 0 ? extractCols(input.test.rows) : [];

  const { y } = dfToXY(input.train, targetCol);
  engine.fit(subTrain, y);
  const newSubTrain = engine.transform(subTrain);
  const newSubTest = subTest.length > 0 ? engine.transform(subTest) : [];

  const sameSize = newSubTrain[0]?.length === colIndices.length;
  let newColumns: string[];
  let stitchTrain: any[][];
  let stitchTest: any[][];

  if (sameSize) {
    newColumns = [...input.train.columns];
    stitchTrain = input.train.rows.map((row, r) => {
      const nr = [...row];
      for (let i = 0; i < colIndices.length; i++) nr[colIndices[i]] = newSubTrain[r][i];
      return nr;
    });
    stitchTest = input.test.rows.map((row, r) => {
      const nr = [...row];
      if (newSubTest[r]) for (let i = 0; i < colIndices.length; i++) nr[colIndices[i]] = newSubTest[r][i];
      return nr;
    });
  } else {
    const keepIdx = input.train.columns.map((_, i) => i).filter(i => !colIndices.includes(i));
    const keptNames = keepIdx.map(i => input.train.columns[i]);
    const newNames = Array.from({ length: newSubTrain[0]?.length ?? 0 }, (_, i) => `${node.type}_${i}`);
    newColumns = [...keptNames, ...newNames];
    stitchTrain = input.train.rows.map((row, r) => [...keepIdx.map(i => row[i]), ...newSubTrain[r]]);
    stitchTest = input.test.rows.map((row, r) => [...keepIdx.map(i => row[i]), ...(newSubTest[r] ?? [])]);
  }

  const features = newColumns.filter(c => c !== targetCol);
  const newMeta: DataPacketMeta = {
    ...input.meta,
    featureColumns: features,
    schema: { ...input.meta.schema, columns: features.map(n => ({ name: n, type: 'numeric' as const })), shape: [stitchTrain.length, features.length] },
  };

  let fittedParams: Record<string, any> = {};
  if (node.type === 'StandardScaler') fittedParams = { means: engine.means, stds: engine.stds };
  else if (node.type === 'MinMaxScaler') fittedParams = { mins: engine.mins, maxs: engine.maxs };
  else if (node.type === 'SimpleImputer') fittedParams = { values: engine.values, strategy: engine.strategy };
  else if (node.type === 'DropColumns') fittedParams = { droppedColumns: selectedCols };

  return {
    packet: { train: { columns: newColumns, rows: stitchTrain }, test: { columns: newColumns, rows: stitchTest }, meta: newMeta },
    engine, fittedParams,
  };
}

// ─── Execute Model node ──────────────────────────────────────────────────────

async function executeModel(
  node: PipelineNode, input: DataPacket, onProgress?: (msg: string) => void, signal?: AbortSignal
): Promise<{ packet: DataPacket; engine: any; fitted: Record<string, unknown>; predictions: number[] }> {
  const cfg = node.config as any;
  const targetCol = input.meta.targetColumn;

  // If no split done, auto-split 80/20
  let trainDF = input.train;
  let testDF = input.test;
  if (testDF.rows.length === 0 && trainDF.rows.length > 0) {
    const { train, test } = splitData(trainDF, 0.2, true);
    trainDF = train;
    testDF = test;
  }

  const { X: xTrain, y: yTrain } = dfToXY(trainDF, targetCol);
  const { X: xTest } = dfToXY(testDF, targetCol);

  if (xTrain.length === 0) throw new Error('No training data available for model');

  const taskType = input.meta.taskType === 'auto' ? detectTaskType(yTrain) : input.meta.taskType;
  const isReg = taskType === 'regression';

  let engine: any;
  const p = cfg.params || {};

  switch (node.type) {
    case 'LinearRegression':
      engine = new TFLinearRegressionEngine(p.learningRate ?? 0.01, p.epochs ?? 100, p.optimizer ?? 'adam', p.batchSize ?? 32);
      break;
    case 'LogisticRegression':
      engine = new TFLogisticRegressionEngine(p.learningRate ?? 0.01, p.epochs ?? 100, p.optimizer ?? 'adam', p.batchSize ?? 32);
      break;
    case 'MLPClassifier':
      engine = new TFMLPEngine(p.hiddenLayers ?? '8,4', p.learningRate ?? 0.05, p.epochs ?? 100, p.optimizer ?? 'adam', p.batchSize ?? 32);
      break;
    case 'KNN':
      engine = new KNNEngine(p.k ?? 5, isReg);
      break;
    case 'DecisionTree':
      engine = new DecisionTreeEngine(p.maxDepth ?? 5, p.minSamplesSplit ?? 2, isReg, p.criterion ?? 'gini');
      break;
    case 'RandomForest':
      engine = new RandomForestEngine(p.nEstimators ?? 20, p.maxDepth ?? 5, isReg);
      break;
    case 'SVM':
      engine = new SVMEngine(p.C ?? 1);
      break;
    case 'NaiveBayes':
      engine = new NaiveBayesEngine();
      break;
    case 'KMeans':
      engine = new KMeansEngine(p.nClusters ?? 3, p.maxIterations ?? 50);
      break;
    default:
      throw new Error(`Unknown model type: ${node.type}`);
  }

  await engine.fit(xTrain, yTrain, onProgress, signal);

  const predictions = engine.predict(xTest.length > 0 ? xTest : xTrain);
  const probabilities = engine.predictProba ? engine.predictProba(xTest.length > 0 ? xTest : xTrain) : undefined;

  const outPacket: DataPacket = {
    train: trainDF, test: testDF, meta: input.meta,
    predictions, probabilities: Array.isArray(probabilities?.[0]) ? probabilities : undefined,
  };

  return { packet: outPacket, engine, fitted: {}, predictions };
}

// ─── Main Execution Pass ──────────────────────────────────────────────────────

export async function executePipeline(
  pipeline: Pipeline,
  onNodeStart: (id: string) => void,
  onNodeDone: (id: string, result: NodeExecutionResult) => void,
  onProgress: (msg: string) => void,
  signal?: AbortSignal,
): Promise<{ results: Record<string, NodeExecutionResult>; fittedEngines: Record<string, any>; featureNames: string[]; nodePackets: Record<string, DataPacket> }> {
  const { nodes, edges } = pipeline;
  const order = topologicalSort(nodes, edges);
  if (!order) throw new Error('Pipeline has a cycle');

  const nodeMap = new Map(nodes.map(n => [n.id, n]));
  const results: Record<string, NodeExecutionResult> = {};
  const fittedEngines: Record<string, any> = {};
  // Per-node isolated data packets
  const nodePackets: Record<string, DataPacket> = {};

  for (const nodeId of order) {
    if (signal?.aborted) throw new Error('Training cancelled by user');
    const node = nodeMap.get(nodeId)!;
    onNodeStart(nodeId);
    onProgress(`Processing: ${node.label}…`);
    await sleep(30);

    try {
      let result: NodeExecutionResult = { nodeId, success: true };

      // ─── DATASET ────────────────────────────────────────────────
      if (node.family === 'dataset') {
        const packet = buildDatasetPacket(node);
        nodePackets[nodeId] = packet;
        result.outputSchema = packet.meta.schema;
      }

      // ─── SPLIT ──────────────────────────────────────────────────
      else if (node.family === 'split') {
        const input = getInputPacket(nodeId, edges, nodePackets);
        const packet = executeSplit(node, input);
        nodePackets[nodeId] = packet;
        onProgress(`  Split: ${packet.train.rows.length} train / ${packet.test.rows.length} test`);
      }

      // ─── PREPROCESSOR ──────────────────────────────────────────
      else if (node.family === 'preprocessor') {
        const input = getInputPacket(nodeId, edges, nodePackets);
        const { packet, engine, fittedParams } = executePreprocessor(node, input);
        nodePackets[nodeId] = packet;
        if (engine) fittedEngines[nodeId] = engine;
        result.fittedParams = fittedParams;
      }

      // ─── MODEL ─────────────────────────────────────────────────
      // ─── STACKER ─────────────────────────────────────────────
      else if (node.family === 'stacker') {
        const parentEdges = edges.filter(e => e.target === nodeId);
        const parentPackets = parentEdges.map(e => nodePackets[e.source]).filter(Boolean);
        if (parentPackets.length === 0) throw new Error('Stacker has no model inputs');
        const cfg = node.config as any;
        const passthrough = cfg.passthrough ?? false;

        // Collect predictions from each parent model as columns
        const basePkt = parentPackets[0];
        const targetCol = basePkt.meta.targetColumn;
        const tIdx = basePkt.test.columns.indexOf(targetCol);
        const yTest = tIdx >= 0 ? basePkt.test.rows.map(r => Number(r[tIdx])) : [];
        const nTest = basePkt.test.rows.length;
        const nTrain = basePkt.train.rows.length;

        // Stack train predictions (re-predict on train set for stacking)
        const stackTrainCols: number[][] = [];
        const stackTestCols: number[][] = [];
        const predColNames: string[] = [];
        for (let pi = 0; pi < parentEdges.length; pi++) {
          const srcId = parentEdges[pi].source;
          const eng = fittedEngines[srcId];
          const pkt = nodePackets[srcId];
          if (!eng || !pkt) continue;
          const srcTarget = pkt.meta.targetColumn;
          const srcFeatIdx = pkt.train.columns.map((c, i) => c !== srcTarget ? i : -1).filter(i => i !== -1);
          const xTr = pkt.train.rows.map(row => srcFeatIdx.map(i => Number(row[i])));
          const xTe = pkt.test.rows.map(row => srcFeatIdx.map(i => Number(row[i])));
          const trainPreds = eng.predict(xTr);
          const testPreds = eng.predict(xTe);
          stackTrainCols.push(trainPreds);
          stackTestCols.push(testPreds);
          const srcNode = nodeMap.get(srcId);
          predColNames.push(srcNode ? `pred_${srcNode.label.replace(/\s+/g,'_')}` : `pred_${pi}`);
        }

        let stackTrainRows: any[][];
        let stackTestRows: any[][];
        let stackColumns: string[];

        if (passthrough) {
          // Include original features + stacked predictions
          const origFeats = basePkt.meta.featureColumns;
          const origFeatIdx = origFeats.map(c => basePkt.train.columns.indexOf(c)).filter(i => i !== -1);
          stackColumns = [...origFeats, ...predColNames, targetCol];
          stackTrainRows = Array.from({ length: nTrain }, (_, r) => [
            ...origFeatIdx.map(i => basePkt.train.rows[r][i]),
            ...stackTrainCols.map(col => col[r] ?? 0),
            tIdx >= 0 ? basePkt.train.rows[r][tIdx] : 0,
          ]);
          stackTestRows = Array.from({ length: nTest }, (_, r) => [
            ...origFeatIdx.map(i => basePkt.test.rows[r]?.[i] ?? 0),
            ...stackTestCols.map(col => col[r] ?? 0),
            yTest[r] ?? 0,
          ]);
        } else {
          stackColumns = [...predColNames, targetCol];
          stackTrainRows = Array.from({ length: nTrain }, (_, r) => [
            ...stackTrainCols.map(col => col[r] ?? 0),
            tIdx >= 0 ? basePkt.train.rows[r][tIdx] : 0,
          ]);
          stackTestRows = Array.from({ length: nTest }, (_, r) => [
            ...stackTestCols.map(col => col[r] ?? 0),
            yTest[r] ?? 0,
          ]);
        }

        const stackFeatures = stackColumns.filter(c => c !== targetCol);
        nodePackets[nodeId] = {
          train: { columns: stackColumns, rows: stackTrainRows },
          test: { columns: stackColumns, rows: stackTestRows },
          meta: {
            targetColumn: targetCol,
            taskType: basePkt.meta.taskType,
            featureColumns: stackFeatures,
            schema: { columns: stackFeatures.map(n => ({ name: n, type: 'numeric' as const })), shape: [stackTrainRows.length, stackFeatures.length], target: targetCol, taskType: basePkt.meta.taskType === 'auto' ? undefined : basePkt.meta.taskType },
          },
        };
        onProgress(`  Stacked ${parentEdges.length} model predictions → ${stackFeatures.length} features`);
      }

      // ─── MODEL ─────────────────────────────────────────────────
      else if (node.family === 'model') {
        const input = getInputPacket(nodeId, edges, nodePackets);
        const { packet, engine, predictions } = await executeModel(node, input, onProgress, signal);
        nodePackets[nodeId] = packet;
        fittedEngines[nodeId] = engine;
        result.predictions = predictions;
      }

      // ─── EVALUATION ────────────────────────────────────────────
      else if (node.family === 'evaluation') {
        const input = getInputPacket(nodeId, edges, nodePackets);
        const preds = input.predictions ?? [];
        const tIdx = input.test.columns.indexOf(input.meta.targetColumn);
        const yTest = tIdx >= 0 ? input.test.rows.map(r => Number(r[tIdx])) : [];
        const aligned = yTest.slice(0, preds.length);

        if (preds.length > 0 && aligned.length > 0) {
          result.metrics = node.type === 'RegressionMetrics'
            ? computeRegressionMetrics(aligned, preds)
            : computeClassificationMetrics(aligned, preds);
        } else {
          result.metrics = {};
        }
        nodePackets[nodeId] = input;
      }

      // ─── OUTPUT ────────────────────────────────────────────────
      else if (node.family === 'output') {
        const input = getInputPacket(nodeId, edges, nodePackets);
        if (node.type === 'CSVExport' && input.predictions?.length) {
          const csvStr = 'prediction\n' + input.predictions.join('\n');
          const blob = new Blob([csvStr], { type: 'text/csv' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url; a.download = (node.config as any).filename ?? 'predictions.csv'; a.click();
          URL.revokeObjectURL(url);
        }
        nodePackets[nodeId] = input;
      }

      // Build snapshots
      const packet = nodePackets[nodeId];
      if (packet) {
        result.inputSnapshot = {
          shape: [packet.train.rows.length, packet.train.columns.length],
          sample: packet.train.rows.slice(0, 5),
          featureNames: [...packet.train.columns],
        };
        
        let rawPacket = packet;
        const splitNode = pipeline.nodes.find(n => n.family === 'split');
        if (splitNode && nodePackets[splitNode.id]) {
          rawPacket = nodePackets[splitNode.id];
        } else {
          const dsNode = pipeline.nodes.find(n => n.family === 'dataset');
          if (dsNode && nodePackets[dsNode.id]) {
            rawPacket = nodePackets[dsNode.id];
          }
        }
        
        // Ensure we match the predictions with the correct raw rows
        // If predictions were evaluated on xTest, it has the same length as packet.test.rows
        const isTest = packet.test.rows.length > 0;
        const outRows = isTest ? rawPacket.test.rows : rawPacket.train.rows;
        const outCols = isTest ? rawPacket.test.columns : rawPacket.train.columns;

        result.outputSnapshot = result.predictions
          ? { 
              shape: [result.predictions.length, outCols.length + 1], 
              sample: outRows.slice(0, 5).map((r, i) => [...r, result.predictions![i]]), 
              featureNames: [...outCols, 'prediction'] 
            }
          : result.inputSnapshot;
      }

      results[nodeId] = result;
      onNodeDone(nodeId, result);
    } catch (e: any) {
      const err: NodeExecutionResult = { nodeId, success: false, error: e.message };
      results[nodeId] = err;
      onNodeDone(nodeId, err);
      throw e;
    }
  }

  const lastPacket = Object.values(nodePackets).pop();
  const featureNames = lastPacket?.meta?.featureColumns ?? [];
  return { results, fittedEngines, featureNames, nodePackets };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getInputPacket(nodeId: string, edges: PipelineEdge[], packets: Record<string, DataPacket>): DataPacket {
  const parentEdges = edges.filter(e => e.target === nodeId);
  if (parentEdges.length === 0) throw new Error('Node has no input connection');
  // For single-input nodes, return the first (and only) parent packet
  const parentPacket = packets[parentEdges[0].source];
  if (!parentPacket) throw new Error('Input node has not been processed yet');
  return parentPacket;
}

function sleep(ms: number) { return new Promise(r => setTimeout(r, ms)); }

export function runInference(
  pipeline: Pipeline,
  inputRecord: Record<string, number>,
  fittedEngines: Record<string, any>,
  nodePackets: Record<string, DataPacket>
): { value: number; rawValue: string | number } | null {
  const { nodes, edges } = pipeline;
  const order = topologicalSort(nodes, edges);
  if (!order) return null;
  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  // Store intermediate rows for each node
  const rows: Record<string, any[]> = {};

  for (const nodeId of order) {
    const node = nodeMap.get(nodeId);
    if (!node) continue;
    if (node.family === 'dataset') {
       const packet = nodePackets[nodeId];
       if (!packet) continue;
       const columns = packet.train.columns;
       const row = columns.map(c => inputRecord[c] ?? 0);
       rows[nodeId] = row;
    } else if (node.family === 'split' || node.family === 'output' || node.family === 'evaluation') {
       const parentEdge = edges.find(e => e.target === nodeId);
       if (parentEdge) rows[nodeId] = rows[parentEdge.source];
    } else if (node.family === 'preprocessor') {
       const parentEdge = edges.find(e => e.target === nodeId);
       if (!parentEdge || !rows[parentEdge.source]) continue;
       const inputRow = rows[parentEdge.source];
       
       const engine = fittedEngines[nodeId];
       if (!engine) { rows[nodeId] = inputRow; continue; }
       
       const cfg = node.config as any;
       const packet = nodePackets[parentEdge.source];
       const targetCol = packet.meta.targetColumn;
       
       const selectedCols: string[] = cfg.targetColumns || [];
       let colIndices: number[];
       if (selectedCols.length > 0) {
         colIndices = selectedCols.map((c: string) => packet.train.columns.indexOf(c)).filter((i: number) => i !== -1);
       } else {
         colIndices = packet.train.columns.map((_, i) => i).filter(i => packet.train.columns[i] !== targetCol);
       }
       
       if (colIndices.length === 0) { rows[nodeId] = inputRow; continue; }
       
       const subRow = colIndices.map(i => inputRow[i]);
       const newSubRow = engine.transform([subRow])[0];
       
       const sameSize = newSubRow.length === colIndices.length;
       if (sameSize) {
         const outRow = [...inputRow];
         for (let i = 0; i < colIndices.length; i++) outRow[colIndices[i]] = newSubRow[i];
         rows[nodeId] = outRow;
       } else {
         const keepIdx = packet.train.columns.map((_, i) => i).filter(i => !colIndices.includes(i));
         rows[nodeId] = [...keepIdx.map(i => inputRow[i]), ...newSubRow];
       }
    } else if (node.family === 'stacker') {
       // Stacker: run all parent model engines and stack predictions
       const parentEdges = edges.filter(e => e.target === nodeId);
       const cfg = node.config as any;
       const passthrough = cfg.passthrough ?? false;
       const stackedValues: number[] = [];
       let origRow: any[] | null = null;
       let origPacket: DataPacket | null = null;

       for (const pe of parentEdges) {
         const eng = fittedEngines[pe.source];
         const parentRow = rows[pe.source];
         const pkt = nodePackets[pe.source];
         if (!eng || !parentRow || !pkt) continue;
         const tc = pkt.meta.targetColumn;
         const fi = pkt.train.columns.map((c, i) => c !== tc ? i : -1).filter(i => i !== -1);
         const xRow = [fi.map(i => parentRow[i])];
         stackedValues.push(eng.predict(xRow)[0]);
         if (!origRow) { origRow = parentRow; origPacket = pkt; }
       }

       if (passthrough && origRow && origPacket) {
         const tc = origPacket.meta.targetColumn;
         const fi = origPacket.train.columns.map((c, i) => c !== tc ? i : -1).filter(i => i !== -1);
         rows[nodeId] = [...fi.map(i => origRow![i]), ...stackedValues, 0];
       } else {
         rows[nodeId] = [...stackedValues, 0];
       }
    } else if (node.family === 'model') {
       const parentEdge = edges.find(e => e.target === nodeId);
       if (!parentEdge || !rows[parentEdge.source]) continue;
       
       const engine = fittedEngines[nodeId];
       if (!engine) continue;
       
       const inputRow = rows[parentEdge.source];
       const packet = nodePackets[parentEdge.source];
       if (!packet) continue;
       const targetCol = packet.meta.targetColumn;
       
       const featuresIdx = packet.train.columns.map((c, i) => c !== targetCol ? i : -1).filter(i => i !== -1);
       const X = [featuresIdx.map(i => inputRow[i])];
       
       const pred = engine.predict(X)[0];
       // Store row for downstream nodes (stacker might need it)
       rows[nodeId] = inputRow;
       // Check if this is the last model (no outgoing to stacker)
       const outEdges = edges.filter(e => e.source === nodeId);
       const feedsStacker = outEdges.some(e => {
         const tgt = nodeMap.get(e.target);
         return tgt?.family === 'stacker';
       });
       
       if (!feedsStacker) {
         let rawPred: string | number = pred;
         
         // Trace backwards to apply inverse transforms
         let currId = nodeId;
         const visited = new Set<string>();
         while (currId) {
           visited.add(currId);
           const pEdge = edges.find(e => e.target === currId);
           if (!pEdge) break;
           const parentId = pEdge.source;
           const pNode = nodeMap.get(parentId);
           if (!pNode) break;
           if (pNode.family === 'preprocessor') {
             const pEng = fittedEngines[parentId];
             const pCfg = pNode.config as any;
             const pParentEdge = edges.find(e => e.target === parentId);
             if (pParentEdge && pEng && pEng.inverse_transform) {
                const ppPkt = nodePackets[pParentEdge.source];
                if (ppPkt) {
                  const selectedCols: string[] = pCfg.targetColumns || [];
                  const targetColRaw = ppPkt.meta.targetColumn;
                  let colIndices: number[];
                  if (selectedCols.length > 0) {
                    colIndices = selectedCols.map((c: string) => ppPkt.train.columns.indexOf(c)).filter((i: number) => i !== -1);
                  } else {
                    colIndices = ppPkt.train.columns.map((_, i) => i).filter(i => ppPkt.train.columns[i] !== targetColRaw);
                  }
                  
                  const targetIdx = ppPkt.train.columns.indexOf(targetColRaw);
                  const transformedIdx = colIndices.indexOf(targetIdx);
                  if (transformedIdx !== -1) {
                    // This preprocessor modified the target column!
                    const dummyRow = new Array(colIndices.length).fill(0);
                    dummyRow[transformedIdx] = Number(rawPred);
                    const inverted = pEng.inverse_transform([dummyRow])[0];
                    rawPred = inverted[transformedIdx];
                  }
                }
             }
           }
           currId = parentId;
         }
         return { value: pred, rawValue: rawPred };
       }
    }
  }
  return null;
}

// Re-export helpers for backward compat
export function trainTestSplit(X: any[][], y: any[], testSplit = 0.2) {
  const n = X.length;
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const splitIdx = Math.floor(n * (1 - testSplit));
  return {
    X_train: indices.slice(0, splitIdx).map(i => X[i]),
    X_test: indices.slice(splitIdx).map(i => X[i]),
    y_train: indices.slice(0, splitIdx).map(i => y[i]),
    y_test: indices.slice(splitIdx).map(i => y[i]),
  };
}
