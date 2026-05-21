import { useState, useEffect, useRef, useMemo } from 'react';
import Icon from './Icon';
import { getAllModels } from '../../models';
import { usePlayground } from '../../store';

interface SearchItem {
  id: string;
  type: 'model' | 'action' | 'category';
  title: string;
  subtitle: string;
  icon: string;
  payload?: any;
}

interface SearchOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SearchOverlay({ isOpen, onClose }: SearchOverlayProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const { setActiveModel } = usePlayground();

  const allModels = useMemo(() => getAllModels(), []);

  const items = useMemo((): SearchItem[] => {
    const results: SearchItem[] = [];

    // Add Models
    allModels.forEach(m => {
      results.push({
        id: `model-${m.id}`,
        type: 'model',
        title: m.name,
        subtitle: `${m.category} · ${m.shortName}`,
        icon: m.categoryIcon || 'analytics',
        payload: m.id
      });
    });

    // Add Actions/Shortcuts
    results.push({ id: 'action-reset', type: 'action', title: 'Reset Viewport', subtitle: 'Reset canvas zoom and position', icon: 'restart_alt', payload: 'reset-view' });
    results.push({ id: 'action-train', type: 'action', title: 'Start/Stop Training', subtitle: 'Toggle training loop', icon: 'play_arrow', payload: 'toggle-train' });

    if (!query) return results.slice(0, 8);

    const q = query.toLowerCase();
    return results.filter(item => 
      item.title.toLowerCase().includes(q) || 
      item.subtitle.toLowerCase().includes(q)
    );
  }, [allModels, query]);

  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
          e.preventDefault();
          onClose(); // This should be onOpen but we use a toggle logic in parent
        }
        return;
      }

      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % items.length);
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + items.length) % items.length);
      }
      if (e.key === 'Enter') {
        const item = items[selectedIndex];
        if (item) handleSelect(item);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, items, selectedIndex, onClose]);

  const handleSelect = (item: SearchItem) => {
    if (item.type === 'model') {
      setActiveModel(item.payload);
    }
    // Handle other actions here if needed
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="search-overlay" onClick={onClose}>
      <div className="search-modal" onClick={e => e.stopPropagation()}>
        <div className="search-modal__header">
          <Icon name="search" size={20} className="search-modal__icon" />
          <input
            ref={inputRef}
            className="search-modal__input"
            placeholder="Search models, actions, or datasets... (⌘K)"
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
          <div className="search-modal__esc">ESC</div>
        </div>

        <div className="search-modal__body">
          {items.length > 0 ? (
            <div className="search-results">
              {items.map((item, idx) => (
                <div
                  key={item.id}
                  className={`search-item ${idx === selectedIndex ? 'search-item--active' : ''}`}
                  onClick={() => handleSelect(item)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                >
                  <div className="search-item__icon-wrap">
                    <Icon name={item.icon} size={18} />
                  </div>
                  <div className="search-item__content">
                    <div className="search-item__title">{item.title}</div>
                    <div className="search-item__subtitle">{item.subtitle}</div>
                  </div>
                  {idx === selectedIndex && (
                    <div className="search-item__hint">Enter ↵</div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="search-modal__empty">
              <Icon name="search_off" size={48} />
              <p>No results found for "{query}"</p>
            </div>
          )}
        </div>

        <div className="search-modal__footer">
          <div className="search-modal__shortcut">
            <span>↑↓</span> to navigate
          </div>
          <div className="search-modal__shortcut">
            <span>↵</span> to select
          </div>
          <div className="search-modal__shortcut">
            <span>ESC</span> to close
          </div>
        </div>
      </div>
    </div>
  );
}
