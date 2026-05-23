/**
 * NodeInspector — right panel: Config · Preview · Interface · Theory tabs
 * Provides beautiful visual previews (confusion matrices, coefficients, trees) and educational theory modules.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePipeline } from '../PipelineStore';
import { getCatalogEntry, FAMILY_META } from '../NodeCatalog';
import type { PipelineNode } from '../PipelineTypes';
import Icon from '../../components/common/Icon';

type Tab = 'config' | 'preview' | 'interface' | 'theory';

export interface TheoryInfo {
  definition: string;
  math: string;
  workflow: string[];
  hyperparams: Record<string, string>;
  tips: string[];
}

export const THEORY_CATALOG: Record<string, TheoryInfo> = {
  Dataset: {
    definition: 'Loads a structured CSV dataset and acts as the entrypoint for the pipeline. It handles automatic column parsing and holdout splits.',
    math: 'Dataset = X (Features) ∪ y (Target Variable)',
    workflow: [
      'Parse the raw CSV rows into continuous floats.',
      'Identify the selected Target Column and separate it from feature columns.',
      'Perform holdout train/test split or K-fold partitioning using a deterministic shuffle.'
    ],
    hyperparams: {
      'Target Column': 'The label or column you want the models to predict.',
      'Task Type': 'Regression for continuous numbers; Classification for categories.',
      'Split Ratio': 'The percentage of data set aside to evaluate the model (typically 20%).'
    },
    tips: [
      'Ensure the target column matches your goal (e.g. classification target has discrete 0 or 1 values).',
      'Clean or pre-process columns before feeding them to downstream preprocessors.'
    ]
  },
  StandardScaler: {
    definition: 'Normalizes feature scales by centering to mean zero and scaling to unit variance. Essential for distance-based ML models.',
    math: 'z = (x - μ) / σ',
    workflow: [
      'Compute the mean (μ) and standard deviation (σ) for each feature column on the training split.',
      'Subtract μ and divide by σ for every value in training and testing splits.',
      'Guarantees output features are centered at 0 with standard deviation of 1.'
    ],
    hyperparams: {
      'withMean': 'Centering features to zero mean.',
      'withStd': 'Scaling features to unit variance.'
    },
    tips: [
      'Always use this preprocessor before distance-based estimators like KNN, SVM, or PCA.',
      'Without scaling, features with large raw values (like Salary) will completely dominate distance metrics over small values (like Age).'
    ]
  },
  MinMaxScaler: {
    definition: 'Scales all features to fall strictly within a specific bounded range (typically 0 to 1). Highly useful for neural networks.',
    math: 'x_scaled = (x - x_min) / (x_max - x_min)',
    workflow: [
      'Determine the minimum and maximum values for each column on the training set.',
      'Linearly transform values to map between 0 and 1.',
      'Ensures no outliers skew downstream computations out of bounds.'
    ],
    hyperparams: {
      'featureRange': 'Defines the upper and lower bounds of the output (defaults to [0, 1]).'
    },
    tips: [
      'Highly effective for Neural Networks (MLP) or algorithms expecting positive, bounded values.',
      'Sensitive to outliers: a single extreme maximum will compress normal values into a very tight, uninformative interval.'
    ]
  },
  SimpleImputer: {
    definition: 'Replaces missing values (NaN) in your feature columns using a computed statistical strategy.',
    math: 'x_imputed = strategy(X_column)',
    workflow: [
      'Scan the dataset for null, undefined, or missing values.',
      'Compute the Mean, Median, or Mode of the column on non-null values.',
      'Fill all blank entries in that column with the calculated placeholder.'
    ],
    hyperparams: {
      'Strategy': 'Mean (average), Median (middle value), or Most Frequent (mode).'
    },
    tips: [
      'Always place this node immediately after your Dataset node if you expect missing data.',
      'Most ML algorithms will crash or fail if blank missing values reach their training loops.'
    ]
  },
  PCA: {
    definition: 'Principal Component Analysis. Projects high-dimensional data onto orthogonal vectors of maximum variance, reducing dimensionality.',
    math: 'Cov = X_centered^T * X_centered / (N - 1)',
    workflow: [
      'Center the dataset to zero mean.',
      'Construct a covariance matrix comparing all feature columns.',
      'Compute eigenvectors (Principal Components) of the covariance matrix using power iteration.',
      'Project feature coordinates onto the top K eigenvectors.'
    ],
    hyperparams: {
      'Components': 'Number of principal components to output. Each represents a combination of features maximizing variance.'
    },
    tips: [
      'Scale your features using StandardScaler before applying PCA; variance calculations are highly sensitive to raw scale.',
      'Use PCA to visualize high-dimensional data in 2D or 3D, or to combat the "curse of dimensionality".'
    ]
  },
  LinearRegression: {
    definition: 'Ordinary Least Squares regression. Models the relationship between features and a continuous target variable using a linear line.',
    math: 'y = w_1·x_1 + w_2·x_2 + ... + w_n·x_n + b',
    workflow: [
      'Initialize coefficients (weights) and intercept (bias) to zero.',
      'Predict targets, calculate residuals (error = pred - actual).',
      'Compute gradients and adjust weights via gradient descent: w = w - lr * gradient.',
      'Iterate until weights converge.'
    ],
    hyperparams: {
      'learningRate': 'Determines the size of steps taken towards the optimal weights. Too high causes oscillations; too low makes training extremely slow.',
      'maxIterations': 'Maximum gradient descent update steps.'
    },
    tips: [
      'Use exclusively for continuous regression tasks (e.g. predicting price, temperature).',
      'Ensure standard scaling is applied beforehand to help gradient descent converge faster.'
    ]
  },
  LogisticRegression: {
    definition: 'A linear classification model that outputs probabilities mapped between 0 and 1 using the sigmoid function.',
    math: 'p = 1 / (1 + e^-z) where z = w·x + b',
    workflow: [
      'Map linear combination of weights and inputs into the Sigmoid activation function.',
      'Compute probability of binary class membership (0 or 1).',
      'Optimize weights using cross-entropy loss gradients.',
      'Threshold predictions at 0.5 for final binary classification.'
    ],
    hyperparams: {
      'learningRate': 'Controls step size in optimization.',
      'maxIterations': 'The maximum training updates.'
    },
    tips: [
      'Perfect baseline for binary or multiclass classification.',
      'Inspect the coefficients (weights) in the Preview tab to see which features most strongly drive predictions!'
    ]
  },
  KNN: {
    definition: 'K-Nearest Neighbors. A non-parametric classifier/regressor that predicts based on the majority vote of the K closest training points.',
    math: 'd(p, q) = √Σ(p_i - q_i)²',
    workflow: [
      'Store all training feature points and their target labels (instance-based learning).',
      'For any new test point, calculate the Euclidean distance to every stored point.',
      'Sort points by distance, isolate the K nearest neighbors.',
      'Perform a majority vote (classification) or average (regression) to generate predictions.'
    ],
    hyperparams: {
      'K Neighbors': 'Number of adjacent voting points. Lower K creates complex decision boundaries (overfitting); higher K creates smooth boundaries (underfitting).'
    },
    tips: [
      'Feature scaling is absolutely mandatory before using KNN.',
      'Can be slow to predict on large datasets because it must calculate distances to every single training instance at inference time.'
    ]
  },
  DecisionTree: {
    definition: 'A flow-chart-like recursive tree structure that splits data at decision nodes using feature thresholds that maximize Gini or Entropy reduction.',
    math: 'Gini = 1 - Σ (p_i)^2',
    workflow: [
      'Evaluate all feature thresholds to find the split that maximizes impurity reduction (Information Gain).',
      'Split the dataset into left (<= threshold) and right (> threshold) child groups.',
      'Recursively split children until max depth is reached or node sample counts are too small.',
      'Assign the majority class of the leaf node as the prediction.'
    ],
    hyperparams: {
      'maxDepth': 'The maximum levels of the tree. Deep trees overfit; shallow trees underfit.',
      'criterion': 'Gini (CART) or Entropy (ID3/C4.5) to measure node purity.'
    },
    tips: [
      'Does not require feature scaling: splitting rules are invariant to linear scaling.',
      'Provides high interpretability. Check the Preview tab after training to view the tree hierarchy!'
    ]
  },
  RandomForest: {
    definition: 'An ensemble bagging algorithm that trains multiple independent decision trees on bootstrapped subsets and averages their votes.',
    math: 'RF = MajorityVote(Tree_1, Tree_2, ... Tree_B)',
    workflow: [
      'Create B bootstrapped training subsets by sampling rows with replacement.',
      'Train an independent Decision Tree on each subset.',
      'For each split in a tree, only consider a random subset of features.',
      'Vote (classification) or average (regression) predictions across all trees.'
    ],
    hyperparams: {
      'nEstimators': 'Number of trees in the forest. More trees improve generalization but take longer to train.',
      'maxDepth': 'Depth limit of individual trees.'
    },
    tips: [
      'Extremely robust, handles outliers, missing values, and high dimensions effortlessly.',
      'One of the best default classifiers for complex tabular data.'
    ]
  },
  SVM: {
    definition: 'Support Vector Machine. Maximizes the margin boundary between two classes. Projects linear splits or maps features into higher dimensions using kernels.',
    math: 'f(x) = sign(w^T x + b)',
    workflow: [
      'Identify support vectors: points sitting directly on the margin borders.',
      'Solve quadratic optimization to find parameters w and b maximizing the margin.',
      'Classify testing points based on which side of the decision boundary they land.'
    ],
    hyperparams: {
      'C (Penalty)': 'Regularization parameter. Large C penalizes misclassifications (narrow margin, risks overfitting); small C allows margin violations (wider margin, risks underfitting).'
    },
    tips: [
      'Very powerful for clean, high-dimensional datasets with clear margins.',
      'Always apply StandardScaler before training SVM to ensure feature dimensions are weighed equally.'
    ]
  },
  NaiveBayes: {
    definition: 'A probabilistic classifier that applies Bayes Theorem under the "naive" assumption of conditional independence between features.',
    math: 'P(c|x) ∝ P(c) · Π P(x_i|c)',
    workflow: [
      'Compute the prior probability of each class: P(c).',
      'For each class, compute the mean and variance of each feature column.',
      'Evaluate incoming predictions using the Gaussian log probability likelihood formula.',
      'Predict the class that yields the maximum posterior probability.'
    ],
    hyperparams: {},
    tips: [
      'Extremely fast to train and predict, as it only requires a single linear pass over the dataset.',
      'Works surprisingly well even when the "feature independence" assumption is violated.'
    ]
  },
  MLPClassifier: {
    definition: 'Multi-Layer Perceptron. A deep feed-forward artificial neural network consisting of layers of nodes with non-linear activation functions (ReLU) and backpropagation training.',
    math: 'a^[l] = g(W^[l] · a^[l-1] + b^[l])',
    workflow: [
      'Feed inputs forward through hidden layers with ReLU non-linear activation.',
      'Compute output layer probabilities using Sigmoid or Softmax functions.',
      'Calculate cross-entropy error between prediction probabilities and true targets.',
      'Backpropagate errors through layers to calculate weight gradients using chain rule.',
      'Update weights and biases using Gradient Descent to minimize network loss.'
    ],
    hyperparams: {
      'hiddenLayers': 'Comma-separated node counts for each hidden layer (e.g. "8,4" translates to 2 hidden layers).',
      'learningRate': 'Controls backpropagation step size during optimization.',
      'maxEpochs': 'The number of complete sweeps over the training set during training.'
    },
    tips: [
      'Extremely flexible, capable of learning highly complex non-linear decision boundaries.',
      'Always standardize input features (using StandardScaler) to help neural net gradients converge smoothly!'
    ]
  },
  KMeans: {
    definition: 'K-Means Clustering. An unsupervised clustering algorithm that groups unlabeled multi-dimensional data points into K cohesive clusters centered around moving centroids.',
    math: 'J = Σ_i Σ_k r_ik ||x_i - μ_k||²',
    workflow: [
      'Initialize K cluster centroids randomly from the dataset.',
      'Assign each sample to its nearest centroid based on Euclidean distance.',
      'Recompute centroids by averaging the coordinates of all samples assigned to that cluster.',
      'Repeat assignments and centroid updates iteratively until centroids stabilize (convergence).'
    ],
    hyperparams: {
      'nClusters': 'The number of cluster groupings (K) to partition.',
      'maxIterations': 'Maximum centroid optimization iterations if convergence is not reached.'
    },
    tips: [
      'Highly sensitive to initial centroid coordinates. Runs multiple times or uses k-means++ internally.',
      'Does not require target label columns for clustering since it is an unsupervised learning node!'
    ]
  },
  ClassificationMetrics: {
    definition: 'Evaluation node. Compares predictions against true targets on classification tasks to measure accuracy, precision, recall, and F1 score.',
    math: 'Accuracy = (TP + TN) / (TP + TN + FP + FN)',
    workflow: [
      'Align predictions from the fitted model with actual target labels on the testing split.',
      'Count True Positives, False Positives, True Negatives, and False Negatives.',
      'Formulate accuracy, precision (exactness), recall (completeness), and F1-score (harmonic mean).'
    ],
    hyperparams: {},
    tips: [
      'Look at F1 score instead of Accuracy if your classes are highly imbalanced.',
      'Inspect the Confusion Matrix in the Preview tab to see which classes are being confused.'
    ]
  },
  RegressionMetrics: {
    definition: 'Evaluation node. Measures regression predictions against true targets using continuous error metrics.',
    math: 'MSE = (1/N) · Σ (y_pred - y_true)²',
    workflow: [
      'Align predictions with continuous targets.',
      'Compute residual differences.',
      'Evaluate Mean Squared Error (MSE), Root MSE, Mean Absolute Error (MAE), and R-squared coefficient.'
    ],
    hyperparams: {},
    tips: [
      'R-squared measures the proportion of variance explained. 1.0 is a perfect fit; 0.0 means the model is as simple as predicting the average.',
      'RMSE is in the same units as the target column, making it highly intuitive to communicate.'
    ]
  },
  DropColumns: {
    definition: 'Feature Selection. Drop Columns structurally removes non-informative or redundant features from the data flow to reduce dimensionality and noise.',
    math: 'X_new = X \\ {c_1, c_2, ...}',
    workflow: [
      'Identify target column to preserve under all conditions.',
      'Select feature columns to be excluded from the dataset.',
      'Filter out the selected columns structurally, updating both the feature names and the underlying data matrices.'
    ],
    hyperparams: {},
    tips: [
      'Dropping highly correlated or redundant features helps prevent multicollinearity issues in linear regression models.',
      'Keep your dataset compact to avoid the curse of dimensionality and speed up model training.'
    ]
  },
  StackPredictions: {
    definition: 'Ensemble node. Collects predictions from multiple base models and stacks them as new feature columns to train a meta-learner (stacked generalization).',
    math: 'X_meta = [f₁(X), f₂(X), ..., fₖ(X)]',
    workflow: [
      'Receive trained predictions from each connected base model.',
      'Create a new feature matrix where each column is one model\'s prediction.',
      'Optionally pass through original features alongside stacked predictions (passthrough mode).',
      'Feed the stacked features to a downstream meta-model for final prediction.'
    ],
    hyperparams: {
      'Passthrough': 'If enabled, original features are included alongside stacked predictions, giving the meta-model access to both raw and predicted features.'
    },
    tips: [
      'Stacking works best when base models are diverse (e.g., mixing tree-based and linear models).',
      'Use a simple meta-model (like Linear Regression) to avoid overfitting the stacked features.',
      'For best results, use cross-validated predictions for stacking instead of direct train predictions.'
    ]
  }
};

export default function NodeInspector() {
  const { pipeline, selectedNodeId, updateNodeConfig, removeNode, loadCSVToNode, nodeResults, featureNames, rawFeatureNames, testPrediction } = usePipeline();
  const node = pipeline.nodes.find(n => n.id === selectedNodeId) ?? null;
  const [tab, setTab] = useState<Tab>('config');

  // Listen for context-menu tab switch events
  useEffect(() => {
    const onPreview = () => setTab('preview');
    const onConfig = () => setTab('config');
    window.addEventListener('pipeline:preview-tab', onPreview);
    window.addEventListener('pipeline:config-tab', onConfig);
    return () => {
      window.removeEventListener('pipeline:preview-tab', onPreview);
      window.removeEventListener('pipeline:config-tab', onConfig);
    };
  }, []);

  // Auto-switch to preview when a trained node is selected
  useEffect(() => {
    if (node && node.state === 'trained' && nodeResults[node.id]) {
      setTab('preview');
    } else if (node) {
      setTab('config');
    }
  }, [selectedNodeId]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!node) return <InspectorEmpty pipeline={pipeline} />;

  const result = nodeResults[node.id];

  return (
    <div className="pipeline-inspector">
      {/* Node header */}
      <NodeHeader node={node} removeNode={removeNode} />

      {/* Tabs */}
      <div className="pipeline-inspector__tabs">
        {(['config', 'preview', 'interface', 'theory'] as Tab[]).map(t => (
          <button
            key={t}
            className={`pipeline-inspector__tab ${tab === t ? 'pipeline-inspector__tab--active' : ''}`}
            onClick={() => setTab(t)}
            style={t === 'theory' ? { display: 'inline-flex', alignItems: 'center', justifyContent: 'center' } : undefined}
          >
            {t === 'theory' ? <Icon name="info" size={14} /> : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {tab === 'config' && (() => {
        // Trace upstream through edges to find the source Dataset node
        const findUpstreamDataset = (nodeId: string, visited = new Set<string>()): any | null => {
          if (visited.has(nodeId)) return null;
          visited.add(nodeId);
          const n = pipeline.nodes.find(nd => nd.id === nodeId);
          if (!n) return null;
          if (n.family === 'dataset') return n;
          // Walk incoming edges
          const inEdges = pipeline.edges.filter(e => e.target === nodeId);
          for (const edge of inEdges) {
            const found = findUpstreamDataset(edge.source, visited);
            if (found) return found;
          }
          return null;
        };
        const isDataset = node.family === 'dataset';
        const upstreamDS = isDataset ? node : findUpstreamDataset(node.id);
        const dsCfg = upstreamDS?.config as any;
        const allColumns: string[] = dsCfg?.csvHeaders || [];
        const targetCol = dsCfg?.targetColumn || '';

        // For preprocessors (except DropColumns): show ALL columns (including target) so user can choose
        // For models and DropColumns: show only feature columns (exclude target)
        const snapshotCols = result?.inputSnapshot?.featureNames ?? [];
        let activeFeatures: string[];
        if (snapshotCols.length > 0) {
          activeFeatures = (node.family === 'preprocessor' && node.type !== 'DropColumns') ? snapshotCols : snapshotCols.filter((n: string) => n !== targetCol);
        } else if (allColumns.length > 0) {
          activeFeatures = (node.family === 'preprocessor' && node.type !== 'DropColumns') ? allColumns : allColumns.filter((h: string) => h !== targetCol);
        } else {
          activeFeatures = featureNames.length ? (node.type === 'DropColumns' ? featureNames.filter((n: string) => n !== targetCol) : featureNames) : [];
        }

        return <NodeConfig node={node} updateNodeConfig={updateNodeConfig} loadCSVToNode={loadCSVToNode} featureNames={activeFeatures} targetColumn={targetCol} />;
      })()}

      {tab === 'preview' && (
        <NodePreview result={result} node={node} />
      )}

      {tab === 'interface' && (
        <NodeInterface
          isTrained={pipeline.state === 'trained'}
          featureNames={rawFeatureNames}
          testPrediction={testPrediction}
          pipeline={pipeline}
        />
      )}

      {tab === 'theory' && (
        <NodeTheory node={node} />
      )}
    </div>
  );
}

// ── Node Header ───────────────────────────────────────────────────────────────

function NodeHeader({ node, removeNode }: { node: PipelineNode; removeNode: (id: string) => void }) {
  const color = FAMILY_META[node.family]?.color ?? '#888';
  return (
    <div className="pipeline-inspector__node-header" style={{ borderColor: color }}>
      <div>
        <div className="pipeline-inspector__node-label">{node.label}</div>
        <div className="pipeline-inspector__node-family" style={{ color }}>
          {node.family.toUpperCase()} · {node.type}
        </div>
      </div>
      <button className="pipeline-inspector__delete" onClick={() => removeNode(node.id)} title="Delete node">
        <Icon name="delete" size={14} />
      </button>
    </div>
  );
}

// ── Empty State ───────────────────────────────────────────────────────────────

function InspectorEmpty({ pipeline }: any) {
  const allMetrics = pipeline.metrics ?? {};

  return (
    <div className="pipeline-inspector">
      <div className="pipeline-inspector__empty-header">
        <div className="pipeline-inspector__empty-title">Pipeline Inspector</div>
        <div className="pipeline-inspector__empty-sub">Select a node to configure it</div>
      </div>

      <div className="pipeline-inspector__section">
        <div className="pipeline-inspector__section-title">Quick Start</div>
        <div style={{ fontSize: 10, color: 'var(--c-on-surface-variant)', lineHeight: 1.6, opacity: 0.7 }}>
          1. Drag a <strong>Dataset</strong> node onto the canvas<br />
          2. Upload a CSV file in the Dataset node config<br />
          3. Add preprocessors and models<br />
          4. Connect nodes and click <strong>Run Pipeline</strong>
        </div>
      </div>

      {Object.keys(allMetrics).length > 0 && (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Pipeline Metrics</div>
          <div className="pipeline-inspector__metrics">
            {Object.entries(allMetrics).map(([k, v]) => (
              <div key={k} className="pipeline-inspector__metric">
                <span className="pipeline-inspector__metric-label">{k}</span>
                <span className="pipeline-inspector__metric-value">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Config Tab ────────────────────────────────────────────────────────────────

function NodeConfig({ node, updateNodeConfig, loadCSVToNode, featureNames, targetColumn }: {
  node: PipelineNode; updateNodeConfig: (id: string, cfg: any) => void;
  loadCSVToNode: (nodeId: string, text: string, filename: string) => void;
  featureNames: string[]; targetColumn?: string;
}) {
  const catalog = getCatalogEntry(node.family, node.type);
  const cfg = node.config as any;
  const fileRef = useRef<HTMLInputElement>(null);

  const set = useCallback((key: string, value: any) => updateNodeConfig(node.id, { [key]: value }), [node.id, updateNodeConfig]);
  const setParam = useCallback((key: string, value: any) => {
    updateNodeConfig(node.id, { params: { ...(cfg.params ?? {}), [key]: value } });
  }, [node.id, cfg.params, updateNodeConfig]);

  const handleCSVUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => loadCSVToNode(node.id, ev.target?.result as string, file.name);
    reader.readAsText(file);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      {catalog?.description && (
        <div className="pipeline-inspector__desc">{catalog.description}</div>
      )}

      <div className={`pipeline-inspector__state pipeline-inspector__state--${node.state}`}>
        {node.state === 'trained' && '✓ '}{node.state === 'error' && '✗ '}
        {node.state.charAt(0).toUpperCase() + node.state.slice(1)}
      </div>

      {/* DATASET NODE — Upload CSV + Configure */}
      {node.family === 'dataset' && (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Dataset Upload</div>
          <div
            className={`pipeline-inspector__drop-zone ${cfg.csvFilename ? 'pipeline-inspector__drop-zone--loaded' : ''}`}
            onClick={() => fileRef.current?.click()}
          >
            <input ref={fileRef} type="file" accept=".csv" onChange={handleCSVUpload} style={{ display: 'none' }} />
            {cfg.csvFilename ? (
              <>
                <Icon name="check_circle" size={20} style={{ color: '#10b981' }} />
                <div className="pipeline-inspector__drop-loaded">{cfg.csvRowCount ?? 0} rows · {cfg.csvHeaders?.length ?? 0} cols</div>
                <div className="pipeline-inspector__drop-hint">{cfg.csvFilename}</div>
              </>
            ) : (
              <>
                <Icon name="upload_file" size={24} />
                <div className="pipeline-inspector__drop-label">Upload CSV</div>
                <div className="pipeline-inspector__drop-hint">Click to browse</div>
              </>
            )}
          </div>

          {cfg.csvHeaders?.length > 0 && (
            <>
              <div style={{ marginTop: 12 }} />
              <div className="pipeline-inspector__section-title">Configuration</div>
              <Field label="Target Column" tooltip="The column models will predict.">
                <select className="pipeline-inspector__select" value={cfg.targetColumn ?? ''} onChange={e => set('targetColumn', e.target.value)}>
                  <option value="">Select target…</option>
                  {cfg.csvHeaders.map((h: string) => <option key={h} value={h}>{h}</option>)}
                </select>
              </Field>
              <Field label="Task Type" tooltip="Auto-detect classification vs regression.">
                <select className="pipeline-inspector__select" value={cfg.taskType ?? 'auto'} onChange={e => set('taskType', e.target.value)}>
                  <option value="auto">Auto-detect</option>
                  <option value="classification">Classification</option>
                  <option value="regression">Regression</option>
                </select>
              </Field>
            </>
          )}
        </div>
      )}

      {/* SPLIT NODE */}
      {node.family === 'split' && (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Split Configuration</div>
          <Field label={`Test Split: ${Math.round((cfg.testSplit ?? 0.2) * 100)}%`} tooltip="Proportion of the dataset kept for testing.">
            <input type="range" min={0.05} max={0.5} step={0.05} value={cfg.testSplit ?? 0.2} onChange={e => set('testSplit', +e.target.value)} className="pipeline-inspector__slider" />
          </Field>
          <Field label="Stratified" tooltip="Ensures class proportions are balanced across splits.">
            <Toggle checked={cfg.stratified ?? true} onChange={v => set('stratified', v)} />
          </Field>
          <Field label="Shuffle" tooltip="Randomly shuffle data before splitting.">
            <Toggle checked={cfg.shuffle !== false} onChange={v => set('shuffle', v)} />
          </Field>
        </div>
      )}

      {/* PREPROCESSOR */}
      {node.family === 'preprocessor' && cfg.params && (
        <>
          {featureNames.length > 0 && (
            <div className="pipeline-inspector__section">
              <div className="pipeline-inspector__section-title">
                {node.type === 'DropColumns' ? 'Columns to Drop' : 'Target Columns'}
              </div>
              <div style={{ fontSize: 9, opacity: 0.8, marginBottom: 8, lineHeight: 1.3 }}>
                {node.type === 'DropColumns'
                  ? 'Check the specific columns you want to permanently remove.'
                  : 'Check the columns to apply this transformation to. (If none selected, applies to all)'}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 160, overflowY: 'auto', background: 'rgba(0,0,0,0.2)', padding: '8px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.05)' }}>
                {featureNames.map((h: string) => {
                  const isActive = cfg.targetColumns?.includes(h);
                  const isTarget = h === targetColumn;
                  return (
                    <label key={h} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, cursor: 'pointer', color: isActive ? '#fff' : 'var(--c-on-surface-variant)' }}>
                      <input
                        type="checkbox"
                        checked={isActive ?? false}
                        onChange={(e) => {
                          const current = cfg.targetColumns || [];
                          const next = e.target.checked ? [...current, h] : current.filter((c: string) => c !== h);
                          set('targetColumns', next.length > 0 ? next : undefined);
                        }}
                        style={{ cursor: 'pointer', accentColor: 'var(--c-primary)' }}
                      />
                      {h}
                      {isTarget && <span style={{ fontSize: 9, padding: '1px 4px', background: 'var(--c-primary-dark)', color: 'var(--c-primary)', borderRadius: 3, marginLeft: 'auto' }}>TARGET</span>}
                    </label>
                  );
                })}
              </div>
            </div>
          )}
          <div className="pipeline-inspector__section">
            <div className="pipeline-inspector__section-title">Parameters</div>
            {node.type === 'SimpleImputer' && (
              <Field label="Strategy" tooltip="Statistical strategy used to calculate missing numbers.">
                <select className="pipeline-inspector__select" value={cfg.params.strategy ?? 'mean'} onChange={e => setParam('strategy', e.target.value)}>
                  <option value="mean">Mean (Average)</option>
                  <option value="median">Median (Middle value)</option>
                </select>
              </Field>
            )}
            {node.type === 'OneHotEncoder' && (
              <Field label="Drop First" tooltip="Drop the first encoded column to prevent multicollinearity.">
                <Toggle checked={cfg.params.dropFirst ?? false} onChange={v => setParam('dropFirst', v)} />
              </Field>
            )}
          </div>
        </>
      )}

      {/* MODEL */}
      {node.family === 'model' && (() => {
        const tCol = targetColumn || 'Not Set';

        return (
          <>
            <div className="pipeline-inspector__section">
              <div className="pipeline-inspector__section-title">Data Flow (X → y)</div>
              <div style={{ fontSize: 10, background: 'rgba(0,0,0,0.15)', padding: 10, borderRadius: 6, lineHeight: 1.4 }}>
                <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center' }}>
                  <strong style={{ color: '#f59e0b', width: 65, flexShrink: 0 }}>Predicts (y):</strong>
                  <span style={{ padding: '2px 6px', background: 'rgba(245, 158, 11, 0.1)', color: '#fcd34d', border: '1px solid rgba(245, 158, 11, 0.2)', borderRadius: 4, fontFamily: 'monospace' }}>
                    {tCol}
                  </span>
                </div>
                <div style={{ display: 'flex' }}>
                  <strong style={{ color: '#10b981', width: 65, flexShrink: 0 }}>Inputs (X):</strong>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {featureNames.length > 0 ? featureNames.map((f, i) => (
                      <span key={i} style={{ padding: '2px 6px', background: 'rgba(16, 185, 129, 0.1)', color: '#6ee7b7', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: 4, fontFamily: 'monospace' }}>
                        {f}
                      </span>
                    )) : <span style={{ opacity: 0.6 }}>No inputs flowing to this node.</span>}
                  </div>
                </div>
              </div>
              <div style={{ fontSize: 9, opacity: 0.6, marginTop: 8, textAlign: 'center', fontStyle: 'italic' }}>
                To change the target, configure the <strong>Dataset</strong> node.<br />
                To change inputs, use a <strong>Drop Columns</strong> preprocessor.
              </div>
            </div>

            <div className="pipeline-inspector__section">
              <div className="pipeline-inspector__section-title">Hyperparameters</div>
              {cfg.params && Object.entries(cfg.params).map(([k, v]) => (
                <Field key={k} label={k} tooltip={`Modify hyperparameter: ${k}`}>
                  {typeof v === 'number' ? (
                    <input type="number" className="pipeline-inspector__input pipeline-inspector__input--small"
                      value={v as number} step={v < 1 ? 0.001 : 1} onChange={e => setParam(k, +e.target.value)} />
                  ) : typeof v === 'boolean' ? (
                    <Toggle checked={v as boolean} onChange={val => setParam(k, val)} />
                  ) : (
                    <input type="text" className="pipeline-inspector__input pipeline-inspector__input--small"
                      value={String(v)} onChange={e => setParam(k, e.target.value)} />
                  )}
                </Field>
              ))}
            </div>



            {node.type === 'MLPClassifier' && cfg.params?.hiddenLayers && (
              <div className="pipeline-inspector__section">
                <div className="pipeline-inspector__section-title">Network Architecture</div>
                <div style={{ fontSize: 9, opacity: 0.8 }}>
                  Input Features: <strong>{featureNames.length || cfg.trainingSchema?.shape?.[1] || 2}</strong>
                </div>
                <MlpDiagram
                  hiddenLayers={typeof cfg.params.hiddenLayers === 'string'
                    ? cfg.params.hiddenLayers.split(',').map((s: string) => parseInt(s.trim())).filter((n: number) => !isNaN(n))
                    : (cfg.params.hiddenLayers as number[])}
                  inputFeatures={featureNames.length || cfg.trainingSchema?.shape?.[1] || 2}
                />
              </div>
            )}
          </>
        );
      })()}



      {/* STACKER (ENSEMBLE) */}
      {node.family === 'stacker' && (
        <>
          <div className="pipeline-inspector__section">
            <div className="pipeline-inspector__section-title">Stacking Configuration</div>
            <Field label="Passthrough" tooltip="Include original features alongside stacked predictions for the meta-model.">
              <Toggle checked={cfg.passthrough ?? false} onChange={v => updateNodeConfig(node.id, { passthrough: v })} />
            </Field>
            <div style={{ padding: '8px 12px', fontSize: 10, color: 'rgba(255,255,255,0.5)', lineHeight: 1.6 }}>
              Connect multiple trained models into this node. Their predictions become new feature columns for a downstream meta-learner.
            </div>
          </div>
        </>
      )}

      {node.error && (
        <div className="pipeline-inspector__error">{node.error}</div>
      )}
    </div>
  );
}

function MlpDiagram({ hiddenLayers, inputFeatures }: { hiddenLayers: number[]; inputFeatures: number }) {
  const maxNodes = 6;
  const clampedInputs = Math.min(inputFeatures, maxNodes);
  const layers = [clampedInputs, ...hiddenLayers.map(l => Math.min(l, maxNodes)), 1];

  const width = 240;
  const height = 120;
  const paddingX = 20;
  const paddingY = 10;

  const layerSpacing = (width - 2 * paddingX) / Math.max(1, layers.length - 1);

  const nodes: { x: number, y: number, layerIdx: number, isOmitted: boolean }[] = [];
  layers.forEach((nodeCount, layerIdx) => {
    const isClamped = (layerIdx === 0 && inputFeatures > maxNodes) || (layerIdx > 0 && layerIdx < layers.length - 1 && hiddenLayers[layerIdx - 1] > maxNodes);

    const x = paddingX + layerIdx * layerSpacing;
    const ySpacing = (height - 2 * paddingY) / Math.max(1, nodeCount);

    for (let i = 0; i < nodeCount; i++) {
      const y = paddingY + (i + 0.5) * ySpacing;
      nodes.push({ x, y, layerIdx, isOmitted: isClamped && i === Math.floor(nodeCount / 2) });
    }
  });

  return (
    <div style={{ background: 'rgba(0,0,0,0.2)', borderRadius: 6, padding: '16px 8px', display: 'flex', justifyContent: 'center', marginTop: 8 }}>
      <svg width={width} height={height} style={{ overflow: 'visible' }}>
        {/* Edges */}
        {nodes.map((n1, i) =>
          nodes.filter(n2 => n2.layerIdx === n1.layerIdx + 1).map((n2, j) => (
            <line key={`${i}-${j}`} x1={n1.x} y1={n1.y} x2={n2.x} y2={n2.y} stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
          ))
        )}
        {/* Nodes */}
        {nodes.map((n, i) => {
          const color = n.layerIdx === 0 ? '#6366f1' : n.layerIdx === layers.length - 1 ? '#10b981' : '#f59e0b';
          if (n.isOmitted) {
            return (
              <g key={i}>
                <circle cx={n.x} cy={n.y - 4} r={1.5} fill={color} opacity={0.6} />
                <circle cx={n.x} cy={n.y} r={1.5} fill={color} opacity={0.6} />
                <circle cx={n.x} cy={n.y + 4} r={1.5} fill={color} opacity={0.6} />
              </g>
            );
          }
          return <circle key={i} cx={n.x} cy={n.y} r={4} fill={color} />;
        })}
      </svg>
    </div>
  );
}

// ── Preview Tab ────────────────────────────────────────────────────────────────

function NodePreview({ result, node }: { result: any; node: PipelineNode }) {
  if (!result) {
    return (
      <div className="pipeline-inspector__section">
        <div className="pipeline-preview__placeholder">
          <Icon name="visibility" size={24} />
          <div>Run the pipeline to see data previews</div>
        </div>
      </div>
    );
  }

  // Preprocessor custom preview panels
  const renderPreprocessorPreview = () => {
    const params = result.fittedParams;
    if (!params) return null;

    if (node.type === 'StandardScaler' && params.means) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Fitted Scaler Parameters</div>
          <div className="pipeline-preview__metrics">
            {params.means.slice(0, 4).map((m: number, idx: number) => {
              const featName = result.inputSnapshot?.featureNames[idx] ?? `Feat ${idx}`;
              return (
                <div key={idx} className="pipeline-inspector__metric" style={{ fontSize: 10 }}>
                  <span className="pipeline-inspector__metric-label">{featName}</span>
                  <span className="pipeline-inspector__metric-value" style={{ opacity: 0.85 }}>
                    μ: {m.toFixed(2)} · σ: {params.stds[idx]?.toFixed(2)}
                  </span>
                </div>
              );
            })}
            {params.means.length > 4 && <div className="pipeline-preview__placeholder" style={{ padding: 4 }}>+ {params.means.length - 4} more features scaled</div>}
          </div>
        </div>
      );
    }

    if (node.type === 'MinMaxScaler' && params.mins) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Fitted MinMaxScaler Bounds</div>
          <div className="pipeline-preview__metrics">
            {params.mins.slice(0, 4).map((minVal: number, idx: number) => {
              const featName = result.inputSnapshot?.featureNames[idx] ?? `Feat ${idx}`;
              return (
                <div key={idx} className="pipeline-inspector__metric" style={{ fontSize: 10 }}>
                  <span className="pipeline-inspector__metric-label">{featName}</span>
                  <span className="pipeline-inspector__metric-value" style={{ opacity: 0.85 }}>
                    [{minVal.toFixed(2)}, {params.maxs[idx]?.toFixed(2)}]
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    if (node.type === 'DropColumns' && params.droppedColumns) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Dropped Columns</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
            {params.droppedColumns.length > 0 ? (
              params.droppedColumns.map((col: string, idx: number) => (
                <span key={idx} className="pipeline-preview__pred-chip" style={{ background: 'rgba(239,68,68,0.15)', color: '#fca5a5', border: '1px solid rgba(239,68,68,0.2)' }}>
                  {col}
                </span>
              ))
            ) : (
              <span style={{ fontSize: 11, opacity: 0.6, fontStyle: 'italic' }}>No columns dropped</span>
            )}
          </div>
        </div>
      );
    }

    if (node.type === 'SimpleImputer' && params.values) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Fitted Imputed Values ({params.strategy})</div>
          <div className="pipeline-preview__metrics">
            {params.values.slice(0, 4).map((val: number, idx: number) => {
              const featName = result.inputSnapshot?.featureNames[idx] ?? `Feat ${idx}`;
              return (
                <div key={idx} className="pipeline-inspector__metric" style={{ fontSize: 10 }}>
                  <span className="pipeline-inspector__metric-label">{featName}</span>
                  <span className="pipeline-inspector__metric-value">Fill: {val.toFixed(3)}</span>
                </div>
              );
            })}
          </div>
        </div>
      );
    }

    if (node.type === 'PCA' && params.explainedVariance) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">PCA Component Variance</div>
          <div className="pipeline-preview__metrics">
            {params.explainedVariance.map((v: number, idx: number) => (
              <div key={idx} className="pipeline-inspector__metric">
                <span className="pipeline-inspector__metric-label">Component {idx + 1}</span>
                <span className="pipeline-inspector__metric-value">Eigenval: {v.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      );
    }

    return null;
  };

  // Model custom previews (coefficients, weights, tree)
  const renderModelPreview = () => {
    const params = result.fittedParams;
    if (!params) return null;

    if ((node.type === 'LinearRegression' || node.type === 'LogisticRegression' || node.type === 'SVM') && params.weights) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Model Coefficients (Weights)</div>
          <div className="pipeline-preview__metrics">
            {params.weights.slice(0, 5).map((w: number, idx: number) => {
              const featName = result.inputSnapshot?.featureNames[idx] ?? `Feature ${idx}`;
              return (
                <div key={idx} className="pipeline-inspector__metric">
                  <span className="pipeline-inspector__metric-label">{featName}</span>
                  <span className="pipeline-inspector__metric-value" style={{ color: w >= 0 ? '#10b981' : '#ef4444', fontWeight: 600 }}>
                    {w >= 0 ? '+' : ''}{w.toFixed(4)}
                  </span>
                </div>
              );
            })}
            {params.weights.length > 5 && <div className="pipeline-preview__placeholder" style={{ padding: 4 }}>+ {params.weights.length - 5} more coefficients</div>}
            <div className="pipeline-inspector__metric" style={{ borderTop: '0.5px solid var(--c-panel-border)', paddingTop: 4, marginTop: 4 }}>
              <span className="pipeline-inspector__metric-label">Intercept (Bias)</span>
              <span className="pipeline-inspector__metric-value">{params.bias?.toFixed(4) ?? '0.0000'}</span>
            </div>
          </div>
        </div>
      );
    }

    if (node.type === 'DecisionTree' && params.root) {
      const renderNodeTree = (n: any, depth: number): any => {
        if (!n) return null;
        if (depth > 2) return <div style={{ paddingLeft: 12, opacity: 0.5, fontSize: 9 }}>[Sub-tree hidden]</div>;
        if (n.isLeaf) {
          return (
            <div style={{ paddingLeft: 12, color: '#10b981', fontStyle: 'italic', fontSize: 9 }}>
              Leaf: Val = {n.value.toFixed ? n.value.toFixed(2) : n.value} (n={n.samples})
            </div>
          );
        }
        const fName = result.inputSnapshot?.featureNames[n.featIdx] ?? `F${n.featIdx}`;
        return (
          <div style={{ paddingLeft: 12, fontSize: 10, borderLeft: '1px dashed rgba(255,255,255,0.1)' }}>
            <div style={{ color: 'var(--c-primary)', fontWeight: 600 }}>{fName} &lt;= {n.threshold.toFixed(2)} ? (n={n.samples})</div>
            {renderNodeTree(n.left, depth + 1)}
            {renderNodeTree(n.right, depth + 1)}
          </div>
        );
      };

      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Decision Splitting Tree View</div>
          <div style={{ background: 'var(--c-surface-container-highest)', padding: 8, borderRadius: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {renderNodeTree(params.root, 0)}
          </div>
        </div>
      );
    }

    if (node.type === 'MLPClassifier' && params.lossHistory) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Neural Network Training Loss Curve</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ fontSize: 9, opacity: 0.8 }}>
              • Hidden Architecture: <strong style={{ color: 'var(--c-primary)' }}>{params.hiddenLayers?.join(' → ')} nodes</strong><br />
              • Learning Rate: <strong>{params.learningRate}</strong> · Total Epochs: <strong>{params.lossHistory.length}</strong>
            </div>
            {/* Draw mini horizontal progress bars for top 5 epochs loss to show descent! */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, background: 'rgba(0,0,0,0.15)', padding: 8, borderRadius: 6 }}>
              <div style={{ fontSize: 8, textTransform: 'uppercase', color: 'var(--c-on-surface-variant)', letterSpacing: '0.05em' }}>Loss Minimization history:</div>
              {params.lossHistory.filter((_: any, idx: number) => idx === 0 || idx === Math.floor(params.lossHistory.length / 2) || idx === params.lossHistory.length - 1).map((loss: number, idx: number) => {
                const epochNum = idx === 0 ? 1 : idx === 1 ? Math.floor(params.lossHistory.length / 2) + 1 : params.lossHistory.length;
                return (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 9 }}>
                    <span style={{ width: 55, opacity: 0.7 }}>Epoch {epochNum}:</span>
                    <div style={{ flex: 1, height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', background: '#f59e0b', width: `${Math.min(100, Math.max(2, loss * 100))}%` }} />
                    </div>
                    <span style={{ width: 45, textAlign: 'right', fontFamily: 'monospace' }}>{loss.toFixed(4)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      );
    }

    if (node.type === 'KMeans' && params.centroids) {
      return (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Centroid Clusters Overview</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ fontSize: 9, opacity: 0.8 }}>
              • Target Clusters (K): <strong>{params.nClusters}</strong>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, background: 'rgba(0,0,0,0.15)', padding: 8, borderRadius: 6 }}>
              <div style={{ fontSize: 8, textTransform: 'uppercase', color: 'var(--c-on-surface-variant)', letterSpacing: '0.05em', marginBottom: 2 }}>Centroid Coordinates:</div>
              {params.centroids.map((center: number[], idx: number) => (
                <div key={idx} style={{ fontSize: 9, borderBottom: idx < params.centroids.length - 1 ? '0.5px solid rgba(255,255,255,0.04)' : 'none', paddingBottom: 2 }}>
                  <span style={{ color: 'var(--c-primary)', fontWeight: 600 }}>Centroid {idx + 1}:</span>
                  <span style={{ fontFamily: 'monospace', marginLeft: 4 }}>
                    [{center.slice(0, 3).map(v => v.toFixed(3)).join(', ')}{center.length > 3 ? '…' : ''}]
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      );
    }

    return null;
  };

  // Custom visual confusion matrix heatmap for Classification Metrics
  const renderEvaluationPreview = () => {
    const metrics = result.metrics;
    if (!metrics || node.type !== 'ClassificationMetrics' || metrics.tp === undefined) return null;

    return (
      <div className="pipeline-inspector__section">
        <div className="pipeline-inspector__section-title">Confusion Matrix Heatmap</div>
        <div style={{ display: 'grid', gridTemplateColumns: '40px 1fr 1fr', gap: 4, width: '100%', maxWidth: 200, margin: '8px auto', textAlign: 'center', fontSize: 10, fontFamily: 'monospace' }}>
          <div></div>
          <div style={{ fontWeight: 600, color: 'var(--c-on-surface-variant)' }}>Pred 0</div>
          <div style={{ fontWeight: 600, color: 'var(--c-on-surface-variant)' }}>Pred 1</div>

          <div style={{ fontWeight: 600, alignSelf: 'center', color: 'var(--c-on-surface-variant)' }}>True 0</div>
          <div style={{ background: '#10b98125', color: '#10b981', border: '1px solid #10b98140', padding: 8, borderRadius: 4 }}>
            <div>{metrics.tn}</div>
            <div style={{ fontSize: 8, opacity: 0.6 }}>TN</div>
          </div>
          <div style={{ background: '#ef444425', color: '#ef4444', border: '1px solid #ef444440', padding: 8, borderRadius: 4 }}>
            <div>{metrics.fp}</div>
            <div style={{ fontSize: 8, opacity: 0.6 }}>FP</div>
          </div>

          <div style={{ fontWeight: 600, alignSelf: 'center', color: 'var(--c-on-surface-variant)' }}>True 1</div>
          <div style={{ background: '#ef444425', color: '#ef4444', border: '1px solid #ef444440', padding: 8, borderRadius: 4 }}>
            <div>{metrics.fn}</div>
            <div style={{ fontSize: 8, opacity: 0.6 }}>FN</div>
          </div>
          <div style={{ background: '#10b98125', color: '#10b981', border: '1px solid #10b98140', padding: 8, borderRadius: 4 }}>
            <div>{metrics.tp}</div>
            <div style={{ fontSize: 8, opacity: 0.6 }}>TP</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
      {node.family === 'dataset' ? (
        <>
          {result.outputSnapshot && (
            <SnapshotTable label="Dataset" snapshot={result.outputSnapshot} color="#6366f1" />
          )}
        </>
      ) : node.family === 'model' ? (
        <>
          {result.inputSnapshot && (
            <SnapshotTable label="Train Data (Fit)" snapshot={result.inputSnapshot} color="#6366f1" />
          )}
          {result.testInputSnapshot && (
            <SnapshotTable label="Test Data (Evaluate)" snapshot={result.testInputSnapshot} color="#f59e0b" />
          )}
          {renderModelPreview()}
          {result.outputSnapshot && (
            <SnapshotTable label="Test Predictions" snapshot={result.outputSnapshot} color="#10b981" />
          )}
        </>
      ) : (
        <>
          {result.inputSnapshot && (
            <SnapshotTable label="Input" snapshot={result.inputSnapshot} color="#6366f1" />
          )}
          {renderPreprocessorPreview()}
          {renderEvaluationPreview()}
          {result.outputSnapshot && (
            <SnapshotTable label="Output" snapshot={result.outputSnapshot} color="#10b981" />
          )}
        </>
      )}

      {result.metrics && Object.keys(result.metrics).length > 0 && (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Metrics</div>
          <div className="pipeline-inspector__metrics">
            {Object.entries(result.metrics).map(([k, v]) => {
              if (['tp', 'fp', 'fn', 'tn'].includes(k)) return null;
              return (
                <div key={k} className="pipeline-inspector__metric">
                  <span className="pipeline-inspector__metric-label">{k}</span>
                  <span className="pipeline-inspector__metric-value">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {result.predictions && result.predictions.length > 0 && (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Predictions (first 5)</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 4 }}>
            {result.predictions.slice(0, 5).map((p: number, i: number) => (
              <span key={i} className="pipeline-preview__pred-chip">{typeof p === 'number' ? p.toFixed(3) : p}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function SnapshotTable({ label, snapshot, color }: { label: string; snapshot: any; color: string }) {
  return (
    <div className="pipeline-inspector__section">
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        <div className="pipeline-inspector__section-title" style={{ marginBottom: 0 }}>{label}</div>
        <span className="pipeline-preview__shape-badge" style={{ borderColor: color, color }}>
          {snapshot.shape[0]} × {snapshot.shape[1]}
        </span>
      </div>
      {snapshot.sample?.length > 0 && (
        <div className="pipeline-preview__table-wrap">
          <table className="pipeline-preview__table">
            <thead>
              <tr>
                {snapshot.featureNames.map((n: string, i: number) => (
                  <th key={i} className="pipeline-preview__th">{n}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {snapshot.sample.slice(0, 5).map((row: any[], ri: number) => (
                <tr key={ri}>
                  {row.map((v: any, ci: number) => (
                    <td key={ci} className="pipeline-preview__td">{typeof v === 'number' ? v.toFixed(3) : String(v)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ── Interface Tab (test predictions) ─────────────────────────────────────────

function NodeInterface({ isTrained, featureNames, testPrediction, pipeline }: {
  isTrained: boolean; featureNames: string[]; testPrediction: (v: Record<string, number>) => { value: number; rawValue: string | number } | null; pipeline: any;
}) {
  const [inputValues, setInputValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<{ value: number; rawValue: string | number } | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const handlePredict = () => {
    const vals: Record<string, number> = {};
    for (const name of featureNames) {
      vals[name] = parseFloat(inputValues[name] ?? '0') || 0;
    }
    const result = testPrediction(vals);
    setPrediction(result);
    setHasRun(true);
  };

  const isClassification = pipeline.schema?.taskType === 'classification' ||
    pipeline.schema?.taskType === 'auto';

  if (!isTrained) {
    return (
      <div className="pipeline-inspector__section">
        <div className="pipeline-preview__placeholder">
          <Icon name="science" size={24} />
          <div>Train the pipeline first to test predictions</div>
        </div>
      </div>
    );
  }

  if (featureNames.length === 0) {
    return (
      <div className="pipeline-inspector__section">
        <div className="pipeline-preview__placeholder">
          <Icon name="input" size={24} />
          <div>No feature names found</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="pipeline-inspector__section">
        <div className="pipeline-inspector__section-title">Single Prediction</div>
        <div className="pipeline-interface__inputs">
          {featureNames.map(name => (
            <div key={name} className="pipeline-interface__input-row">
              <label className="pipeline-interface__input-label" title={name}>
                {name.length > 14 ? name.slice(0, 12) + '…' : name}
              </label>
              <input
                type="number"
                className="pipeline-inspector__input pipeline-inspector__input--small"
                placeholder="0"
                value={inputValues[name] ?? ''}
                onChange={e => setInputValues(prev => ({ ...prev, [name]: e.target.value }))}
              />
            </div>
          ))}
        </div>

        <button
          className="pipeline-interface__predict-btn"
          onClick={handlePredict}
        >
          <Icon name="play_arrow" size={14} />
          Predict
        </button>
      </div>

      {hasRun && (
        <div className="pipeline-inspector__section">
          <div className="pipeline-inspector__section-title">Result</div>
          {prediction !== null ? (
            <div className="pipeline-interface__result">
              <div className="pipeline-interface__result-label">Prediction</div>
              <div className="pipeline-interface__result-value">
                {typeof prediction.value === 'number' ? prediction.value.toFixed(4) : String(prediction.value)}
              </div>
              {prediction.rawValue !== prediction.value && (
                <div style={{ fontSize: 11, color: 'var(--c-primary)', fontWeight: 600, marginTop: 4, background: 'var(--c-primary-dark)', padding: '4px 8px', borderRadius: 4, textAlign: 'center' }}>
                  Decoded: {typeof prediction.rawValue === 'number' ? prediction.rawValue.toFixed(4) : String(prediction.rawValue)}
                </div>
              )}
              {isClassification && prediction.rawValue === prediction.value && (
                <div className="pipeline-interface__result-class">
                  Class {Math.round(prediction.value)}
                </div>
              )}
            </div>
          ) : (
            <div className="pipeline-interface__result-error">Could not generate prediction</div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Theory Tab ───────────────────────────────────────────────────────────────

function NodeTheory({ node }: { node: PipelineNode }) {
  const info = THEORY_CATALOG[node.type];
  if (!info) {
    return (
      <div className="pipeline-inspector__section">
        <div className="pipeline-preview__placeholder">
          <Icon name="help" size={24} />
          <div>No educational content matches this custom node type.</div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0, padding: '8px 14px', fontFamily: 'Inter, sans-serif' }}>
      <div style={{ marginBottom: 12 }}>
        <h4 style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--c-on-surface)', letterSpacing: '0.04em', margin: '8px 0 4px' }}>Concept</h4>
        <p style={{ fontSize: 11, color: 'var(--c-on-surface-variant)', lineHeight: 1.5, opacity: 0.9 }}>{info.definition}</p>
      </div>

      <div style={{ marginBottom: 12 }}>
        <h4 style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--c-on-surface)', letterSpacing: '0.04em', margin: '8px 0 4px' }}>Mathematical Formula</h4>
        <div style={{ background: 'var(--c-surface-container-highest)', padding: '6px 10px', borderRadius: 4, fontFamily: 'monospace', fontSize: 10, color: 'var(--c-primary)', overflowX: 'auto', borderLeft: '2px solid var(--c-primary)' }}>
          {info.math}
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <h4 style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--c-on-surface)', letterSpacing: '0.04em', margin: '8px 0 4px' }}>Operational Flow</h4>
        <ol style={{ paddingLeft: 14, margin: 0, fontSize: 11, color: 'var(--c-on-surface-variant)', display: 'flex', flexDirection: 'column', gap: 4, lineHeight: 1.4 }}>
          {info.workflow.map((step, idx) => (
            <li key={idx} style={{ opacity: 0.85 }}>{step}</li>
          ))}
        </ol>
      </div>

      {Object.keys(info.hyperparams).length > 0 && (
        <div style={{ marginBottom: 12 }}>
          <h4 style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--c-on-surface)', letterSpacing: '0.04em', margin: '8px 0 4px' }}>Hyperparameter Influence</h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {Object.entries(info.hyperparams).map(([name, desc]) => (
              <div key={name} style={{ fontSize: 10, lineHeight: 1.4 }}>
                <strong style={{ color: 'var(--c-on-surface)', fontWeight: 600 }}>{name}</strong>: <span style={{ color: 'var(--c-on-surface-variant)', opacity: 0.8 }}>{desc}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <h4 style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', color: 'var(--c-on-surface)', letterSpacing: '0.04em', margin: '8px 0 4px' }}>Usage Guidelines</h4>
        <ul style={{ paddingLeft: 12, margin: 0, fontSize: 11, color: 'var(--c-on-surface-variant)', display: 'flex', flexDirection: 'column', gap: 4, lineHeight: 1.4 }}>
          {info.tips.map((tip, idx) => (
            <li key={idx} style={{ opacity: 0.85, listStyleType: 'square' }}>{tip}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function Field({ label, children, tooltip }: { label: string; children: React.ReactNode; tooltip?: string }) {
  return (
    <div className="pipeline-inspector__field" title={tooltip}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        <label className="pipeline-inspector__field-label">{label}</label>
        {tooltip && <span style={{ opacity: 0.35, cursor: 'help', fontSize: 10 }}>🛈</span>}
      </div>
      <div className="pipeline-inspector__field-control">{children}</div>
    </div>
  );
}

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="pipeline-inspector__toggle">
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} />
      <span className="pipeline-inspector__toggle-track" />
    </label>
  );
}
