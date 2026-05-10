import { ThemeProvider } from './theme';
import { PlaygroundProvider } from './store';
import { registerKNN, registerLinearRegression, registerPolynomialRegression, registerLogisticRegression } from './models';
import PlaygroundLayout from './components/PlaygroundLayout';

/* ─── Register all models ─── */
registerLinearRegression();
registerPolynomialRegression();
registerLogisticRegression();
registerKNN();
// Future: registerSVM(), etc.

export default function App() {
  return (
    <ThemeProvider>
      <PlaygroundProvider initialModelId="linear-regression">
        <PlaygroundLayout />
      </PlaygroundProvider>
    </ThemeProvider>
  );
}
