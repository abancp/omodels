import Icon from '../common/Icon';
import { useTheme } from '../../theme';
import { useContext } from 'react';
import { PlaygroundContext } from '../../store/PlaygroundStore';

const NAV_TABS = [
  { id: 'playground', label: 'Playground' },
  { id: 'builder', label: 'Builder' },
  { id: 'compare', label: 'Compare' },
  { id: 'datastudio', label: 'Data Studio' }
] as const;

export type ViewType = 'playground' | 'datastudio' | 'builder' | 'compare';

interface TopNavBarProps {
  onSearchClick: () => void;
  activeView: ViewType;
  onViewChange: (view: ViewType) => void;
}

export default function TopNavBar({ onSearchClick, activeView, onViewChange }: TopNavBarProps) {
  const { toggleTheme, isDark } = useTheme();
  const pgCtx = useContext(PlaygroundContext);
  const model = pgCtx?.model ?? null;

  return (
    <header className="topnav" id="topnav">
      <div className="topnav__left">
        <span className="topnav__brand">omodels</span>
        <nav className="topnav__tabs">
          {NAV_TABS.map((tab) => (
            <div
              key={tab.id}
              className={`topnav__tab ${tab.id === activeView ? 'topnav__tab--active' : ''}`}
              onClick={() => {
                if (tab.id === 'playground' || tab.id === 'datastudio') {
                  onViewChange(tab.id);
                }
              }}
              style={{ cursor: (tab.id === 'playground' || tab.id === 'datastudio') ? 'pointer' : 'not-allowed', opacity: (tab.id === 'playground' || tab.id === 'datastudio') ? 1 : 0.5 }}
            >
              <span>{tab.label}</span>
            </div>
          ))}
        </nav>
      </div>

      <div className="topnav__right">
        {model && (
          <div className="topnav__breadcrumb">
            <span>{model.category}</span>
            <Icon name="chevron_right" size={14} />
            <span>{model.name}</span>
          </div>
        )}
        <div className="topnav__divider" />
        <button className="topnav__icon-btn" title="Search (⌘K)" id="btn-search" onClick={onSearchClick}>
          <Icon name="search" size={16} />
        </button>
        <button className="topnav__icon-btn" title="Layout" id="btn-layout">
          <Icon name="grid_view" size={16} />
        </button>
        <button className="topnav__icon-btn" title="Toggle theme" id="btn-theme" onClick={toggleTheme}>
          <Icon name={isDark ? 'light_mode' : 'dark_mode'} size={16} />
        </button>
        <button className="topnav__icon-btn" title="Share" id="btn-share">
          <Icon name="share" size={16} />
        </button>
        <button className="topnav__icon-btn topnav__icon-btn--primary" title="Run" id="btn-run">
          <Icon name="play_arrow" size={16} />
        </button>
      </div>
    </header>
  );
}
