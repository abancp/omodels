/* ═══════════════════════════════════════
   Data Studio — Main Layout
   Wraps the three-panel layout in DataStudioProvider
   ═══════════════════════════════════════ */
import { useState } from 'react';
import TopNavBar, { type ViewType } from './layout/TopNavBar';
import SearchOverlay from './common/SearchOverlay';
import { DataStudioProvider, useDataStudio } from '../datastudio/DataStudioStore';
import LeftPanel from '../datastudio/panels/LeftPanel';
import CenterPanel from '../datastudio/panels/CenterPanel';
import RightPanel from '../datastudio/panels/RightPanel';
import '../datastudio/DataStudio.css';

interface DataStudioLayoutProps {
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
}

/* Loading overlay — renders inside provider so it can read store */
function LoadingOverlay() {
  const { state } = useDataStudio();
  if (!state.loading.active) return null;
  return (
    <div className="ds-loading-overlay">
      <div className="ds-loading-dots">
        <div className="ds-loading-dots__dot" />
        <div className="ds-loading-dots__dot" />
        <div className="ds-loading-dots__dot" />
      </div>
      {state.loading.message && (
        <span className="ds-loading-text">{state.loading.message}</span>
      )}
    </div>
  );
}

function DSWorkspace({ activeView, onViewChange }: DataStudioLayoutProps) {
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const { state } = useDataStudio();
  const isLoading = state.loading.active;

  return (
    <div className="app" id="app-root" style={{ background: 'var(--c-background)' }}>
      <TopNavBar
        onSearchClick={() => setIsSearchOpen(prev => !prev)}
        activeView={activeView}
        onViewChange={onViewChange}
      />

      <div
        className={`workspace ${isLoading ? 'ds-workspace--loading' : ''}`}
        style={{ display: 'flex', height: 'calc(100vh - 40px)', overflow: 'hidden', position: 'relative' }}
      >
        <LeftPanel />
        <CenterPanel />
        <RightPanel />
        <LoadingOverlay />
      </div>

      <SearchOverlay isOpen={isSearchOpen} onClose={() => setIsSearchOpen(false)} />
    </div>
  );
}

export default function DataStudioLayout(props: DataStudioLayoutProps) {
  return (
    <DataStudioProvider>
      <DSWorkspace {...props} />
    </DataStudioProvider>
  );
}
