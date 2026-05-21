import { ThemeProvider } from './theme';
import { PlaygroundProvider } from './store';
import { registerKNN, registerLinearRegression, registerPolynomialRegression, registerLogisticRegression, registerSVM, registerNaiveBayes, registerDecisionTree, registerRandomForest, registerGBM, registerKMeans, registerDBSCAN, registerGMM, registerPerceptron, registerMLP, registerActivations } from './models';
import PlaygroundLayout from './components/PlaygroundLayout';
import DataStudioLayout from './components/DataStudioLayout';

/* ─── Register all models ─── */
registerLinearRegression();
registerPolynomialRegression();
registerLogisticRegression();
registerSVM();
registerKNN();
registerNaiveBayes();
registerDecisionTree();
registerRandomForest();
registerGBM();
registerKMeans();
registerDBSCAN();
registerGMM();
registerPerceptron();
registerMLP();
registerActivations();

import { useState } from 'react';
import type { ViewType } from './components/layout/TopNavBar';

export default function App() {
  const [currentView, setCurrentView] = useState<ViewType>('playground');

  return (
    <ThemeProvider>
      <PlaygroundProvider initialModelId="linear-regression">
        {currentView === 'playground' ? (
          <PlaygroundLayout onViewChange={setCurrentView} activeView={currentView} />
        ) : (
          <DataStudioLayout onViewChange={setCurrentView} activeView={currentView} />
        )}
      </PlaygroundProvider>
    </ThemeProvider>
  );
}
