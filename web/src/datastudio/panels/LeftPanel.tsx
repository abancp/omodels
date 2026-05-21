/* ═══════════════════════════════════════
   Data Studio — Navigation Panel
   ═══════════════════════════════════════ */
import { useDataStudio } from '../DataStudioStore';
import Icon from '../../components/common/Icon';

const MENU_ITEMS = [
  { id: 'transform', label: 'Transform', icon: 'auto_fix_high' },
  { id: 'edit', label: 'Edit Data', icon: 'edit' },
  { id: 'visualize', label: 'Visualize', icon: 'insights' },
  { id: 'split', label: 'Data Split', icon: 'call_split' },
  { id: 'schema', label: 'Schema', icon: 'account_tree' },
] as const;

export default function LeftPanel() {
  const { state, dispatch } = useDataStudio();
  const { activePanel, transforms, working } = state;

  return (
    <aside className="ds-left">
      {/* Header */}
      <div className="ds-left__header">
        <div className="ds-left__header-row">
          <Icon name="storage" size={16} style={{ color: 'var(--c-primary)' }} />
          <span className="ds-left__header-title">Data Studio</span>
        </div>
        <span className="ds-left__header-sub">Preprocessing & EDA</span>
      </div>

      <div className="ds-left__body">
        {/* Navigation */}
        <nav>
          {MENU_ITEMS.map(item => (
            <div
              key={item.id}
              className={`ds-nav-item ${activePanel === item.id ? 'ds-nav-item--active' : ''}`}
              onClick={() => dispatch({ type: 'SET_ACTIVE_PANEL', panel: item.id as any })}
            >
              <Icon name={item.icon} size={18} />
              <span>{item.label}</span>
              {item.id === 'transform' && transforms.length > 0 && (
                <span className="ds-nav-item__badge">{transforms.length}</span>
              )}
            </div>
          ))}
        </nav>

        {/* Pipeline Timeline */}
        <div className="ds-pipeline">
          <div className="ds-pipeline__head">
            <span className="ds-pipeline__label">Pipeline</span>
            <span className="ds-nav-item__badge" style={{ opacity: .6 }}>{transforms.length}</span>
          </div>
          <div className="ds-pipeline__list">
            {transforms.length === 0 ? (
              <div className="ds-pipeline__empty">
                <Icon name="info" size={10} />
                No steps applied
              </div>
            ) : (
              transforms.map((t) => (
                <div key={t.id} className="ds-pipeline__step" style={{ borderLeft: `2px solid var(--c-primary)` }}>
                  <Icon name="check_circle" size={10} style={{ color: 'var(--c-primary)' }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.label}</span>
                  <button className="ds-pipeline__step-remove" onClick={(e) => { e.stopPropagation(); dispatch({ type: 'REMOVE_TRANSFORM', id: t.id }); }}>
                    <Icon name="close" size={10} />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="ds-left__footer">
        <div className="ds-toolbar__info" style={{ padding: '0 4px', marginBottom: 4 }}>
          <span>ROWS: <span className="ds-toolbar__val">{working.length.toLocaleString()}</span></span>
        </div>
        <div className="ds-toolbar__info" style={{ padding: '0 4px' }}>
          <span>SOURCE: <span className="ds-toolbar__val">{state.source.toUpperCase()}</span></span>
        </div>
      </div>
    </aside>
  );
}
