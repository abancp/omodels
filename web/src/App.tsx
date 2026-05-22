import { ThemeProvider } from './theme';
import { PlaygroundProvider } from './store';
import { registerKNN, registerLinearRegression, registerPolynomialRegression, registerLogisticRegression, registerSVM, registerNaiveBayes, registerDecisionTree, registerRandomForest, registerGBM, registerKMeans, registerDBSCAN, registerGMM, registerPerceptron, registerMLP, registerActivations } from './models';
import PlaygroundLayout from './components/PlaygroundLayout';
import DataStudioLayout from './components/DataStudioLayout';
import PipelineLayout from './pipeline/PipelineLayout';
import './pipeline/Pipeline.css';

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

import { useState, useEffect } from 'react';
import type { ViewType } from './components/layout/TopNavBar';

function getViewFromPathname(path: string): ViewType {
  const cleaned = path.replace(/\/$/, '');
  if (cleaned === '/pipeline') return 'pipeline';
  if (cleaned === '/datastudio') return 'datastudio';
  return 'playground';
}

export default function App() {
  const [currentView, setCurrentView] = useState<ViewType>(() =>
    getViewFromPathname(window.location.pathname)
  );

  useEffect(() => {
    const handlePopState = () => {
      setCurrentView(getViewFromPathname(window.location.pathname));
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const handleViewChange = (view: ViewType) => {
    setCurrentView(view);
    const targetPath = view === 'playground' ? '/' : `/${view}`;
    const newUrl = `${targetPath}${window.location.search}`;
    window.history.pushState(null, '', newUrl);
  };

  return (
    <ThemeProvider>
      <PlaygroundProvider initialModelId="linear-regression">
        {currentView === 'playground' ? (
          <PlaygroundLayout onViewChange={handleViewChange} activeView={currentView} />
        ) : currentView === 'pipeline' ? (
          <PipelineLayout onViewChange={handleViewChange} activeView={currentView} />
        ) : (
          <DataStudioLayout onViewChange={handleViewChange} activeView={currentView} />
        )}
      </PlaygroundProvider>
    </ThemeProvider>
  );
}
