/**
 * PlaygroundLayout — assembles the three-column IDE layout:
 * [SideNav] [Canvas + Code] [Controls]
 */

import TopNavBar from './layout/TopNavBar';
import SideNavBar from './layout/SideNavBar';
import CanvasArea from './panels/CanvasArea';
import CodePanel from './panels/CodePanel';
import ControlsPanel from './panels/ControlsPanel';

export default function PlaygroundLayout() {
  return (
    <div className="app" id="app-root">
      <TopNavBar />
      <div className="workspace">
        <SideNavBar />
        <main className="workspace__main">
          <CanvasArea />
          <CodePanel />
        </main>
        <ControlsPanel />
      </div>
    </div>
  );
}
