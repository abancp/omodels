/**
 * PlaygroundLayout — assembles the three-column IDE layout:
 * [SideNav] | [Canvas + Code] | [Controls]
 *
 * Panels can be resized by dragging handles and minimized.
 * Width is managed here via wrapper divs — inner components
 * keep their own class names untouched.
 */

import { useState, useCallback } from 'react';
import TopNavBar from './layout/TopNavBar';
import SideNavBar from './layout/SideNavBar';
import CanvasArea from './panels/CanvasArea';
import CodePanel from './panels/CodePanel';
import ControlsPanel from './panels/ControlsPanel';
import ResizeHandle from './common/ResizeHandle';
import Icon from './common/Icon';

const SIDE_DEFAULT = 200;
const SIDE_MIN = 48;
const SIDE_MAX = 340;

const CTRL_DEFAULT = 240;
const CTRL_MIN = 48;
const CTRL_MAX = 400;

export default function PlaygroundLayout() {
  const [sideW, setSideW] = useState(SIDE_DEFAULT);
  const [ctrlW, setCtrlW] = useState(CTRL_DEFAULT);
  const [sideMin, setSideMin] = useState(false);
  const [ctrlMin, setCtrlMin] = useState(false);

  const handleSideResize = useCallback((delta: number) => {
    setSideW(prev => Math.max(SIDE_MIN, Math.min(SIDE_MAX, prev + delta)));
  }, []);

  const handleCtrlResize = useCallback((delta: number) => {
    setCtrlW(prev => Math.max(CTRL_MIN, Math.min(CTRL_MAX, prev - delta)));
  }, []);

  const toggleSideMin = useCallback(() => {
    setSideMin(prev => {
      if (prev) setSideW(SIDE_DEFAULT);
      return !prev;
    });
  }, []);

  const toggleCtrlMin = useCallback(() => {
    setCtrlMin(prev => {
      if (prev) setCtrlW(CTRL_DEFAULT);
      return !prev;
    });
  }, []);

  const effectiveSideW = sideMin ? SIDE_MIN : sideW;
  const effectiveCtrlW = ctrlMin ? CTRL_MIN : ctrlW;

  return (
    <div className="app" id="app-root">
      <TopNavBar />
      <div className="workspace">
        {/* Side nav wrapper */}
        <div
          className={`panel-slot panel-slot--side ${sideMin ? 'panel-slot--minimized' : ''}`}
          style={{ width: effectiveSideW }}
        >
          <SideNavBar />
          <button
            className="panel-minimize-btn panel-minimize-btn--side"
            onClick={toggleSideMin}
            title={sideMin ? 'Expand sidebar' : 'Minimize sidebar'}
          >
            {sideMin ? <Icon name="chevron_right" size={14} /> : <Icon name="chevron_left" size={14} />}
          </button>
        </div>

        <ResizeHandle
          direction="vertical"
          onResize={handleSideResize}
          onDoubleClick={() => { setSideW(SIDE_DEFAULT); setSideMin(false); }}
        />

        {/* Main canvas + code */}
        <main className="workspace__main">
          <CanvasArea />
          <CodePanel />
        </main>

        <ResizeHandle
          direction="vertical"
          onResize={handleCtrlResize}
          onDoubleClick={() => { setCtrlW(CTRL_DEFAULT); setCtrlMin(false); }}
        />

        {/* Controls panel wrapper */}
        <div
          className={`panel-slot panel-slot--ctrl ${ctrlMin ? 'panel-slot--minimized' : ''}`}
          style={{ width: effectiveCtrlW }}
        >
          <button
            className="panel-minimize-btn panel-minimize-btn--ctrl"
            onClick={toggleCtrlMin}
            title={ctrlMin ? 'Expand controls' : 'Minimize controls'}
          >
            {ctrlMin ? <Icon name="chevron_left" size={14} /> : <Icon name="chevron_right" size={14} />}
          </button>
          <ControlsPanel />
        </div>
      </div>
    </div>
  );
}
