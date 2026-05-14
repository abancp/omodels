import { ThemeProvider } from './theme';
import { PlaygroundProvider } from './store';
import { registerKNN, registerLinearRegression, registerPolynomialRegression, registerLogisticRegression, registerSVM, registerNaiveBayes, registerDecisionTree, registerRandomForest } from './models';
import PlaygroundLayout from './components/PlaygroundLayout';

/* ─── Register all models ─── */
registerLinearRegression();
registerPolynomialRegression();
registerLogisticRegression();
registerSVM();
registerKNN();
registerNaiveBayes();
registerDecisionTree();
registerRandomForest();

export default function App() {
  return (
    <ThemeProvider>
      <PlaygroundProvider initialModelId="linear-regression">
        <PlaygroundLayout />
      </PlaygroundProvider>
    </ThemeProvider>
  );
}
