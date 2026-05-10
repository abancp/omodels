import { useState } from 'react';
import Icon from '../common/Icon';
import { usePlayground } from '../../store';
import { getCategories, getModelsByCategory } from '../../models';

export default function SideNavBar() {
  const { activeModelId, setActiveModel, mode, setMode } = usePlayground();
  const categories = getCategories();
  const [expandedCats, setExpandedCats] = useState<Set<string>>(
    () => new Set(categories.map((c) => c.id))
  );

  const toggleCategory = (catId: string) => {
    setExpandedCats((prev) => {
      const next = new Set(prev);
      if (next.has(catId)) next.delete(catId);
      else next.add(catId);
      return next;
    });
  };

  return (
    <aside className="sidenav" id="sidenav">
      {/* Header */}
      <div className="sidenav__header">
        <div className="sidenav__title">Models</div>
        <div className="sidenav__version">v2.4.0-stable</div>
      </div>

      {/* Model tree */}
      <nav className="sidenav__tree">
        {categories.map((cat) => {
          const models = getModelsByCategory(cat.id);
          const isExpanded = expandedCats.has(cat.id);
          const hasActiveChild = models.some((m) => m.id === activeModelId);

          return (
            <div key={cat.id} className="sidenav__category">
              <div
                className="sidenav__cat-header"
                onClick={() => toggleCategory(cat.id)}
              >
                <Icon name={cat.icon} size={14} />
                <span className="sidenav__cat-label">{cat.name}</span>
                <Icon
                  name={isExpanded ? 'expand_more' : 'chevron_right'}
                  size={14}
                  className="sidenav__cat-arrow"
                />
              </div>

              {isExpanded && (
                <div className="sidenav__items">
                  {models.map((m) => (
                    <div
                      key={m.id}
                      className={`sidenav__item ${m.id === activeModelId ? 'sidenav__item--active' : ''}`}
                      onClick={() => setActiveModel(m.id)}
                    >
                      {m.name}
                    </div>
                  ))}
                </div>
              )}

              {!isExpanded && hasActiveChild && (
                <div className="sidenav__items">
                  {models
                    .filter((m) => m.id === activeModelId)
                    .map((m) => (
                      <div key={m.id} className="sidenav__item sidenav__item--active">
                        {m.name}
                      </div>
                    ))}
                </div>
              )}
            </div>
          );
        })}

        {/* Placeholder categories */}
        {[
          { icon: 'psychology', name: 'Neural Networks' },
          { icon: 'translate', name: 'NLP' },
          { icon: 'visibility', name: 'Computer Vision' },
        ].filter((p) => !categories.find((c) => c.name === p.name)).map((p) => (
          <div key={p.name} className="sidenav__cat-header sidenav__cat-header--placeholder">
            <Icon name={p.icon} size={14} />
            <span className="sidenav__cat-label">{p.name}</span>
            <Icon name="chevron_right" size={14} className="sidenav__cat-arrow" />
          </div>
        ))}
      </nav>

      {/* Footer — Mode toggle */}
      <div className="sidenav__footer">
        <div className="sidenav__mode-toggle" id="mode-toggle">
          <button
            className={`sidenav__mode-btn ${mode === 'basic' ? 'sidenav__mode-btn--active' : ''}`}
            onClick={() => setMode('basic')}
          >
            <Icon name="tune" size={12} />
            Basic
          </button>
          <button
            className={`sidenav__mode-btn ${mode === 'advanced' ? 'sidenav__mode-btn--active' : ''}`}
            onClick={() => setMode('advanced')}
          >
            <Icon name="build" size={12} />
            Advanced
          </button>
        </div>
      </div>
    </aside>
  );
}
