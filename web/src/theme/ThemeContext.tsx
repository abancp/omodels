import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from 'react';
import { darkTheme, lightTheme, type ThemeTokens } from './tokens';

interface ThemeContextValue {
  theme: ThemeTokens;
  toggleTheme: () => void;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function applyThemeCSSVars(theme: ThemeTokens) {
  const root = document.documentElement;
  const c = theme.colors;
  const entries = Object.entries(c) as [string, string][];
  for (const [key, value] of entries) {
    // camelCase → kebab-case
    const cssVar = `--c-${key.replace(/([A-Z])/g, '-$1').toLowerCase()}`;
    root.style.setProperty(cssVar, value);
  }
  root.setAttribute('data-theme', theme.id);
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [isDark, setIsDark] = useState(() => {
    const stored = localStorage.getItem('omodels-theme');
    return stored ? stored === 'dark' : true; // default dark
  });

  const theme = isDark ? darkTheme : lightTheme;

  useEffect(() => {
    applyThemeCSSVars(theme);
    localStorage.setItem('omodels-theme', theme.id);
  }, [theme]);

  const toggleTheme = useCallback(() => setIsDark((d) => !d), []);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, isDark }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}
