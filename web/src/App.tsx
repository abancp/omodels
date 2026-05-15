import { ThemeProvider } from './theme';
import { PlaygroundProvider } from './store';
import { registerKNN, registerLinearRegression, registerPolynomialRegression, registerLogisticRegression, registerSVM, registerNaiveBayes, registerDecisionTree, registerRandomForest, registerGBM, registerKMeans, registerDBSCAN, registerGMM, registerPerceptron, registerMLP, registerActivations } from './models';
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
registerGBM();
registerKMeans();
registerDBSCAN();
registerGMM();
registerPerceptron();
registerMLP();
registerActivations();

export default function App() {
  return (
    <ThemeProvider>
      <PlaygroundProvider initialModelId="linear-regression">
        <PlaygroundLayout />
      </PlaygroundProvider>
    </ThemeProvider>
  );
}
