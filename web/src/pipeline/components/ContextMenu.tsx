/**
 * ContextMenu — reusable context menu for pipeline canvas & nodes
 */

import { useEffect, useRef } from 'react';
import Icon from '../../components/common/Icon';

export interface ContextMenuItem {
  id: string;
  label: string;
  icon?: string;
  shortcut?: string;
  danger?: boolean;
  disabled?: boolean;
  separator?: false;
  children?: ContextMenuItem[];
  action?: () => void;
}

export interface ContextMenuSeparator {
  separator: true;
  id: string;
}

export type ContextMenuEntry = ContextMenuItem | ContextMenuSeparator;

interface ContextMenuProps {
  x: number;
  y: number;
  items: ContextMenuEntry[];
  onClose: () => void;
}

export default function ContextMenu({ x, y, items, onClose }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click or Escape
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('mousedown', handleClick);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleClick);
      document.removeEventListener('keydown', handleKey);
    };
  }, [onClose]);

  // Clamp menu within viewport
  const vpW = window.innerWidth;
  const vpH = window.innerHeight;
  const menuW = 200;
  const menuH = items.length * 30 + 16;
  const cx = x + menuW > vpW ? vpW - menuW - 8 : x;
  const cy = y + menuH > vpH ? vpH - menuH - 8 : y;

  return (
    <div
      ref={menuRef}
      className="ctx-menu"
      style={{ left: cx, top: cy }}
      onContextMenu={e => e.preventDefault()}
    >
      {items.map(item => {
        if ('separator' in item && item.separator) {
          return <div key={item.id} className="ctx-menu__sep" />;
        }
        const mi = item as ContextMenuItem;
        return (
          <ContextMenuItemRow
            key={mi.id}
            item={mi}
            onClose={onClose}
          />
        );
      })}
    </div>
  );
}

function ContextMenuItemRow({ item, onClose }: { item: ContextMenuItem; onClose: () => void }) {
  const hasChildren = item.children && item.children.length > 0;

  return (
    <div
      className={`ctx-menu__item ${item.danger ? 'ctx-menu__item--danger' : ''} ${item.disabled ? 'ctx-menu__item--disabled' : ''} ${hasChildren ? 'ctx-menu__item--has-sub' : ''}`}
      onClick={() => {
        if (item.disabled) return;
        if (!hasChildren) {
          item.action?.();
          onClose();
        }
      }}
    >
      {item.icon && (
        <span className="ctx-menu__item-icon">
          <Icon name={item.icon} size={13} />
        </span>
      )}
      <span className="ctx-menu__item-label">{item.label}</span>
      {item.shortcut && (
        <span className="ctx-menu__item-shortcut">{item.shortcut}</span>
      )}
      {hasChildren && (
        <>
          <span className="ctx-menu__item-arrow">▸</span>
          <div className="ctx-menu ctx-menu--sub">
            {item.children!.map(child => {
              if ('separator' in child && child.separator) {
                return <div key={child.id} className="ctx-menu__sep" />;
              }
              const ci = child as ContextMenuItem;
              return (
                <ContextMenuItemRow key={ci.id} item={ci} onClose={onClose} />
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
