import Icon from '../common/Icon';
import { useTheme } from '../../theme';
import { usePlayground } from '../../store';

const NAV_TABS = ['Playground', 'Builder', 'Compare'] as const;

export default function TopNavBar() {
  const { toggleTheme, isDark } = useTheme();
  const { model } = usePlayground();
  const activeTab = 'Playground'; // only Playground is functional for now

  return (
    <header className="topnav" id="topnav">
      <div className="topnav__left">
        <span className="topnav__brand">omodels</span>
        <nav className="topnav__tabs">
          {NAV_TABS.map((tab) => (
            <div
              key={tab}
              className={`topnav__tab ${tab === activeTab ? 'topnav__tab--active' : ''}`}
            >
              <span>{tab}</span>
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
        <button className="topnav__icon-btn" title="Search" id="btn-search">
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
