/* ═══════════════════════════════════════
   Data Studio — React Context Store
   ═══════════════════════════════════════ */
import React, { createContext, useContext, useReducer, useCallback } from 'react';
import Papa from 'papaparse';
import type { DataState, Transform, TransformKind, SplitConfig, VizType } from './types';
import { parseData, detectSchema, computeStats, applyTransform } from './engine';

type Action =
  | { type: 'IMPORT_START'; fileName: string; source: string }
  | { type: 'IMPORT_SUCCESS'; data: Record<string, any>[] }
  | { type: 'IMPORT_ERROR'; error: string }
  | { type: 'SET_WORKING_DATA'; data: Record<string, any>[]; name: string; source: string }
  | { type: 'APPEND_DATA'; data: Record<string, any>[] }
  | { type: 'ADD_TRANSFORM'; transform: Transform }
  | { type: 'REMOVE_TRANSFORM'; id: string }
  | { type: 'SET_ACTIVE_PANEL'; panel: DataState['activePanel'] }
  | { type: 'SET_VIZ'; viz: VizType }
  | { type: 'SELECT_COLUMN'; col: string | null }
  | { type: 'EDIT_CELL'; rowIdx: number; column: string; value: any }
  | { type: 'ADD_ROW' }
  | { type: 'ADD_COLUMN'; name: string }
  | { type: 'DELETE_ROW'; rowIdx: number }
  | { type: 'DUPLICATE_ROW'; rowIdx: number }
  | { type: 'CLEAR_DATA' }
  | { type: 'DELETE_COLUMN'; name: string }
  | { type: 'SET_SPLIT_CONFIG'; config: Partial<SplitConfig> }
  | { type: 'SET_SPLIT'; split: { train: any[]; test: any[]; val?: any[] } | null }
  | { type: 'SET_LOADING'; active: boolean; message: string }
  | { type: 'BULK_REPLACE'; find: string; replace: string; column: string | null };

const initialState: DataState = {
  raw: [],
  working: [],
  schema: null,
  columnStats: {},
  transforms: [],
  splitConfig: { trainRatio: 0.8, testRatio: 0.2, valRatio: 0, shuffle: true, seed: 42 },
  split: null,
  fileName: '',
  source: '',
  activePanel: 'transform',
  activeViz: 'histogram',
  selectedColumn: null,
  loading: { active: false, message: '' },
  error: null,
};

const GLOBAL_DS_KEY = '__DATASTUDIO_CONTEXT__';
const DataStudioContext = (() => {
  if (!(globalThis as any)[GLOBAL_DS_KEY]) {
    (globalThis as any)[GLOBAL_DS_KEY] = createContext<{
      state: DataState;
      dispatch: React.Dispatch<Action>;
      importFile: (file: File) => Promise<void>;
      addTransform: (kind: TransformKind, params: any, label: string) => void;
      doExport: () => void;
      doSplit: () => void;
    } | null>(null);
  }
  return (globalThis as any)[GLOBAL_DS_KEY] as React.Context<{
    state: DataState;
    dispatch: React.Dispatch<Action>;
    importFile: (file: File) => Promise<void>;
    addTransform: (kind: TransformKind, params: any, label: string) => void;
    doExport: () => void;
    doSplit: () => void;
  } | null>;
})();

function recomputeWorking(raw: Record<string, any>[], transforms: Transform[]) {
  let working = [...raw];
  transforms.filter(t => t.active).forEach(t => {
    working = applyTransform(working, t.kind, t.params);
  });
  const schema = detectSchema(working);
  const stats = computeStats(working, schema);
  return { working, schema, stats };
}

function reducer(state: DataState, action: Action): DataState {
  switch (action.type) {
    case 'IMPORT_START':
      return { ...state, fileName: action.fileName, source: action.source, loading: { active: true, message: 'Parsing data...' } };
    case 'IMPORT_SUCCESS': {
      const { working, schema, stats } = recomputeWorking(action.data, []);
      return { ...state, raw: action.data, working, schema, columnStats: stats, loading: { active: false, message: '' } };
    }
    case 'SET_WORKING_DATA': {
      const { working, schema, stats } = recomputeWorking(action.data, state.transforms);
      return { ...state, raw: action.data, working, schema, columnStats: stats, fileName: action.name, source: action.source, loading: { active: false, message: '' } };
    }
    case 'APPEND_DATA': {
      const newRaw = [...state.raw, ...action.data];
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats, loading: { active: false, message: '' } };
    }
    case 'ADD_TRANSFORM': {
      const newTransforms = [...state.transforms, action.transform];
      const { working, schema, stats } = recomputeWorking(state.raw, newTransforms);
      return { ...state, transforms: newTransforms, working, schema, columnStats: stats };
    }
    case 'REMOVE_TRANSFORM': {
      const newTransforms = state.transforms.filter(t => t.id !== action.id);
      const { working, schema, stats } = recomputeWorking(state.raw, newTransforms);
      return { ...state, transforms: newTransforms, working, schema, columnStats: stats };
    }
    case 'SET_ACTIVE_PANEL': return { ...state, activePanel: action.panel };
    case 'SET_VIZ': return { ...state, activeViz: action.viz };
    case 'SELECT_COLUMN': return { ...state, selectedColumn: action.col };
    case 'EDIT_CELL': {
      const newRaw = [...state.raw];
      newRaw[action.rowIdx] = { ...newRaw[action.rowIdx], [action.column]: action.value };
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats };
    }
    case 'ADD_ROW': {
      const newRow: Record<string, any> = {};
      state.schema?.columns.forEach(c => { newRow[c.name] = ''; });
      const newRaw = [newRow, ...state.raw];
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats };
    }
    case 'ADD_COLUMN': {
      const newRaw = state.raw.map(r => ({ ...r, [action.name]: '' }));
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats };
    }
    case 'DELETE_ROW': {
      const newRaw = state.raw.filter((_, i) => i !== action.rowIdx);
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats };
    }
    case 'DUPLICATE_ROW': {
      const rowToCopy = state.raw[action.rowIdx];
      const newRaw = [...state.raw];
      newRaw.splice(action.rowIdx + 1, 0, { ...rowToCopy });
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats };
    }
    case 'CLEAR_DATA': return { ...state, raw: [], working: [], schema: null, transforms: [], columnStats: {}, fileName: '', source: 'empty' };
    case 'DELETE_COLUMN': {
      const newRaw = state.raw.map(row => {
        const newRow = { ...row };
        delete newRow[action.name];
        return newRow;
      });
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats };
    }
    case 'SET_SPLIT_CONFIG': return { ...state, splitConfig: { ...state.splitConfig, ...action.config } };
    case 'SET_SPLIT': return { ...state, split: action.split };
    case 'SET_LOADING': return { ...state, loading: { active: action.active, message: action.message } };
    case 'BULK_REPLACE': {
      const { find, replace, column } = action;
      const newRaw = state.raw.map(row => {
        const newRow = { ...row };
        Object.keys(newRow).forEach(key => {
          if (!column || column === key) {
            if (String(newRow[key]) === find) newRow[key] = replace;
          }
        });
        return newRow;
      });
      const { working, schema, stats } = recomputeWorking(newRaw, state.transforms);
      return { ...state, raw: newRaw, working, schema, columnStats: stats };
    }
    default: return state;
  }
}

export const DataStudioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(reducer, initialState);

  const importFile = useCallback(async (file: File) => {
    dispatch({ type: 'IMPORT_START', fileName: file.name, source: 'upload' });
    try {
      const data = await parseData(file);
      dispatch({ type: 'IMPORT_SUCCESS', data });
    } catch (e) {
      dispatch({ type: 'IMPORT_ERROR', error: String(e) });
    }
  }, []);

  const addTransform = useCallback((kind: TransformKind, params: any, label: string) => {
    dispatch({ type: 'ADD_TRANSFORM', transform: { id: Math.random().toString(36).substr(2, 9), kind, params, label, active: true } });
  }, []);

  const doExport = useCallback(() => {
    if (state.working.length === 0) return;
    const csv = Papa.unparse(state.working);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', state.fileName.replace(/\.[^/.]+$/, "") + "_processed.csv");
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [state.working, state.fileName]);

  const doSplit = useCallback(() => {
    dispatch({ type: 'SET_LOADING', active: true, message: 'Splitting dataset...' });
    setTimeout(() => {
      const { trainRatio, testRatio, valRatio, shuffle } = state.splitConfig;
      let data = [...state.working];
      if (shuffle) data.sort(() => Math.random() - 0.5);

      const trainIdx = Math.floor(data.length * trainRatio);
      const testIdx = Math.floor(data.length * (trainRatio + testRatio));

      const split = {
        train: data.slice(0, trainIdx),
        test: data.slice(trainIdx, testIdx),
        val: valRatio > 0 ? data.slice(testIdx) : undefined,
      };

      dispatch({ type: 'SET_SPLIT', split });
      dispatch({ type: 'SET_LOADING', active: false, message: '' });
    }, 500);
  }, [state.working, state.splitConfig]);

  return (
    <DataStudioContext.Provider value={{ state, dispatch, importFile, addTransform, doExport, doSplit }}>
      {children}
    </DataStudioContext.Provider>
  );
};

export const useDataStudio = () => {
  const ctx = useContext(DataStudioContext);
  if (!ctx) throw new Error('useDataStudio must be used within DataStudioProvider');
  return ctx;
};
