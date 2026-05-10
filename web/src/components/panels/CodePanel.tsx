/**
 * Code panel — shows code snippets from the active model descriptor.
 * Language tabs switch between Python / C++ / JavaScript.
 * Has a collapse/expand toggle to show/hide.
 */

import { useCallback, type ReactElement } from 'react';
import Icon from '../common/Icon';
import { usePlayground } from '../../store';
import type { CodeLine } from '../../models';

function renderLine(line: CodeLine) {
  if (!line.highlights || line.highlights.length === 0) {
    return <span>{line.text || '\u00A0'}</span>;
  }

  const parts: ReactElement[] = [];
  let cursor = 0;

  const sorted = [...line.highlights].sort((a, b) => a.start - b.start);

  for (const hl of sorted) {
    if (cursor < hl.start) {
      parts.push(<span key={`t${cursor}`}>{line.text.slice(cursor, hl.start)}</span>);
    }
    const className =
      hl.type === 'keyword'
        ? 'code__keyword'
        : hl.type === 'number'
          ? 'code__number'
          : 'code__string';
    parts.push(
      <span key={`h${hl.start}`} className={className}>
        {line.text.slice(hl.start, hl.end)}
      </span>
    );
    cursor = hl.end;
  }

  if (cursor < line.text.length) {
    parts.push(<span key={`r${cursor}`}>{line.text.slice(cursor)}</span>);
  }

  return <>{parts}</>;
}

export default function CodePanel() {
  const { model, activeCodeTab, setActiveCodeTab, isCodePanelOpen, toggleCodePanel } = usePlayground();

  const copyCode = useCallback(() => {
    if (!model) return;
    const snippet = model.codeSnippets[activeCodeTab];
    const text = snippet.lines.map((l) => l.text).join('\n');
    navigator.clipboard.writeText(text);
  }, [model, activeCodeTab]);

  if (!model || model.codeSnippets.length === 0) return null;

  const activeSnippet = model.codeSnippets[activeCodeTab];

  // Collapsed state — just show a thin toggle strip
  if (!isCodePanelOpen) {
    return (
      <div className="codepanel codepanel--collapsed" id="code-panel">
        <button className="codepanel__toggle-btn" onClick={toggleCodePanel} title="Show code">
          <Icon name="code" size={12} />
          <span>Code</span>
          <Icon name="keyboard_arrow_up" size={12} />
        </button>
      </div>
    );
  }

  return (
    <div className="codepanel" id="code-panel">
      {/* Tab bar */}
      <div className="codepanel__tabbar">
        <div className="codepanel__tabs">
          {model.codeSnippets.map((s, i) => (
            <div
              key={s.language}
              className={`codepanel__tab ${i === activeCodeTab ? 'codepanel__tab--active' : ''}`}
              onClick={() => setActiveCodeTab(i)}
            >
              {s.language}
            </div>
          ))}
        </div>
        <div className="codepanel__actions">
          <button className="codepanel__action-btn" title="Copy" onClick={copyCode} id="btn-copy-code">
            <Icon name="content_copy" size={14} />
          </button>
          <button className="codepanel__action-btn" title="Open in new tab" id="btn-open-code">
            <Icon name="open_in_new" size={14} />
          </button>
          <button className="codepanel__action-btn" title="Hide code panel" onClick={toggleCodePanel} id="btn-toggle-code">
            <Icon name="keyboard_arrow_down" size={14} />
          </button>
        </div>
      </div>

      {/* Code content */}
      <div className="codepanel__content">
        {activeSnippet.lines.map((line, i) => (
          <div key={i} className="codepanel__line">
            {renderLine(line)}
          </div>
        ))}
      </div>
    </div>
  );
}
