/**
 * Model Registry — the core abstraction that makes the system modular.
 */

import type { ComponentType } from 'react';

/* ─── Parameter Schema ─── */

export type ParamType = 'slider' | 'select' | 'toggle' | 'chip-select';

/** Parameters can be tagged as basic (default) or advanced */
export type ParamLevel = 'basic' | 'advanced';

export interface SliderParam {
  type: 'slider';
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  formatValue?: (v: number) => string;
  level?: ParamLevel;
}

export interface SelectParam {
  type: 'select';
  key: string;
  label: string;
  options: { value: string; label: string }[];
  defaultValue: string;
  level?: ParamLevel;
}

export interface ToggleParam {
  type: 'toggle';
  key: string;
  label: string;
  defaultValue: boolean;
  level?: ParamLevel;
}

export interface ChipSelectParam {
  type: 'chip-select';
  key: string;
  label: string;
  options: { value: string; label: string; icon?: string }[];
  defaultValue: string;
  level?: ParamLevel;
}

export type ParamDescriptor = SliderParam | SelectParam | ToggleParam | ChipSelectParam;

/* ─── Dataset Schema ─── */

export interface DatasetOption {
  value: string;
  label: string;
  icon?: string;
}

export interface DatasetConfig {
  options: DatasetOption[];
  defaultDataset: string;
  params: ParamDescriptor[];
}

/* ─── Code Snippets ─── */

export interface CodeLine {
  text: string;
  highlights?: { start: number; end: number; type: 'keyword' | 'number' | 'string' }[];
}

export interface CodeSnippet {
  language: string;
  lines: CodeLine[];
}

/* ─── Metric ─── */

export interface MetricValue {
  label: string;
  value: string;
  isPrimary?: boolean;
}

/* ─── Visualization Props ─── */

export interface VisualizationProps {
  params: Record<string, unknown>;
  dataset: string;
  datasetParams: Record<string, unknown>;
  isTraining: boolean;
  /** Incremented when user clicks Reset — visualizations should clear weights/loss/state */
  resetVersion: number;
  onTrainingComplete: () => void;
  onMetricsUpdate: (metrics: MetricValue[]) => void;
}

/* ─── Model Descriptor ─── */

export interface ModelDescriptor {
  id: string;
  name: string;
  shortName: string;
  vizLabel: string;
  category: string;
  categoryIcon: string;
  params: ParamDescriptor[];
  dataset: DatasetConfig;
  codeSnippets: CodeSnippet[];
  defaultMetrics: MetricValue[];
  VisualizationComponent: ComponentType<VisualizationProps>;
  trainable?: boolean;
}

/* ─── Category info ─── */

export interface ModelCategory {
  id: string;
  name: string;
  icon: string;
}

/* ─── Registry ─── */

const registry = new Map<string, ModelDescriptor>();
const categories = new Map<string, ModelCategory>();

export function registerModel(descriptor: ModelDescriptor) {
  registry.set(descriptor.id, descriptor);
  if (!categories.has(descriptor.category)) {
    categories.set(descriptor.category, {
      id: descriptor.category,
      name: descriptor.category,
      icon: descriptor.categoryIcon,
    });
  }
}

export function getModel(id: string): ModelDescriptor | undefined {
  return registry.get(id);
}

export function getAllModels(): ModelDescriptor[] {
  return Array.from(registry.values());
}

export function getCategories(): ModelCategory[] {
  return Array.from(categories.values());
}

export function getModelsByCategory(categoryId: string): ModelDescriptor[] {
  return getAllModels().filter((m) => m.category === categoryId);
}
