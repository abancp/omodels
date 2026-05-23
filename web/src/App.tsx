import { Analytics } from '@vercel/analytics/react';
import { ThemeProvider } from './theme';
import { PlaygroundProvider } from './store';
import { registerKNN, registerLinearRegression, registerPolynomialRegression, registerLogisticRegression, registerSVM, registerNaiveBayes, registerDecisionTree, registerRandomForest, registerGBM, registerKMeans, registerDBSCAN, registerGMM, registerPerceptron, registerMLP, registerActivations } from './models';
import PlaygroundLayout from './components/PlaygroundLayout';
import DataStudioLayout from './components/DataStudioLayout';
import PipelineLayout from './pipeline/PipelineLayout';
import Icon from './components/common/Icon';
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

  const [pendingViewChange, setPendingViewChange] = useState<ViewType | null>(null);

  // Sync back/forward popstate
  useEffect(() => {
    const handlePopState = () => {
      setCurrentView(getViewFromPathname(window.location.pathname));
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  // Sync page unload warning when dirty
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      const isDirty =
        (currentView === 'playground' && (window as any).isPlaygroundDirty?.()) ||
        (currentView === 'pipeline' && (window as any).isPipelineDirty?.()) ||
        (currentView === 'datastudio' && (window as any).isDataStudioDirty?.());

      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [currentView]);

  const handleViewChange = (view: ViewType) => {
    if (view === currentView) return;

    const isDirty =
      (currentView === 'playground' && (window as any).isPlaygroundDirty?.()) ||
      (currentView === 'pipeline' && (window as any).isPipelineDirty?.()) ||
      (currentView === 'datastudio' && (window as any).isDataStudioDirty?.());

    if (isDirty) {
      setPendingViewChange(view);
    } else {
      executeViewChange(view);
    }
  };

  const executeViewChange = (view: ViewType) => {
    setCurrentView(view);
    const targetPath = view === 'playground' ? '/' : `/${view}`;
    const newUrl = `${targetPath}${window.location.search}`;
    window.history.pushState(null, '', newUrl);
  };

  const handleOpenInNewTab = () => {
    if (pendingViewChange) {
      const targetPath = pendingViewChange === 'playground' ? '/' : `/${pendingViewChange}`;
      const targetUrl = `${window.location.origin}${targetPath}${window.location.search}`;
      window.open(targetUrl, '_blank');
      setPendingViewChange(null);
    }
  };

  const handleDiscardAndSwitch = () => {
    if (pendingViewChange) {
      executeViewChange(pendingViewChange);
      setPendingViewChange(null);
    }
  };

  const viewLabels: Record<ViewType, string> = {
    playground: 'Playground',
    pipeline: 'Pipeline',
    datastudio: 'Data Studio'
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

        {/* Tab switch warning modal */}
        {pendingViewChange && (
          <div className="confirm-modal-overlay">
            <div className="confirm-modal-card">
              <div className="confirm-modal-header">
                <div className="confirm-modal-icon">
                  <Icon name="warning" size={20} />
                </div>
                <h3 className="confirm-modal-title">Unsaved Work in Memory</h3>
              </div>
              <p className="confirm-modal-message">
                You have active data or an ongoing session in your current{' '}
                <strong>{viewLabels[currentView]}</strong> view. Switching tabs will discard
                your current workspace state.
                <br />
                <br />
                Would you like to open <strong>{viewLabels[pendingViewChange]}</strong> in a new browser tab/window to keep this session alive, or discard and switch?
              </p>
              <div className="confirm-modal-actions">
                <button
                  className="confirm-btn confirm-btn--secondary"
                  onClick={() => setPendingViewChange(null)}
                >
                  Cancel
                </button>
                <button
                  className="confirm-btn confirm-btn--primary"
                  onClick={handleOpenInNewTab}
                >
                  Open in New Tab
                </button>
                <button
                  className="confirm-btn confirm-btn--danger"
                  onClick={handleDiscardAndSwitch}
                >
                  Discard &amp; Switch
                </button>
              </div>
            </div>
          </div>
        )}
        <Analytics />
      </PlaygroundProvider>
    </ThemeProvider>
  );
}
