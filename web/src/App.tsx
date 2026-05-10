import { ThemeProvider } from './theme';
import { PlaygroundProvider } from './store';
import { registerKNN, registerLinearRegression } from './models';
import PlaygroundLayout from './components/PlaygroundLayout';

/* ─── Register all models ─── */
registerLinearRegression();
registerKNN();
// Future: registerLogisticRegression(), registerSVM(), etc.

export default function App() {
  return (
    <ThemeProvider>
      <PlaygroundProvider initialModelId="linear-regression">
        <PlaygroundLayout />
      </PlaygroundProvider>
    </ThemeProvider>
  );
}
