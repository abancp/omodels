/**
 * Design tokens for the omodels design system.
 * All colors, typography, spacing, and shape tokens are defined here.
 * Dark-first, with a matching light palette.
 */

export interface ColorTokens {
  /* Surfaces – layered depth */
  surface: string;
  surfaceDim: string;
  surfaceBright: string;
  surfaceContainerLowest: string;
  surfaceContainerLow: string;
  surfaceContainer: string;
  surfaceContainerHigh: string;
  surfaceContainerHighest: string;
  surfaceVariant: string;
  surfaceTint: string;

  /* On-surface */
  onSurface: string;
  onSurfaceVariant: string;
  inverseSurface: string;
  inverseOnSurface: string;

  /* Primary */
  primary: string;
  onPrimary: string;
  primaryContainer: string;
  onPrimaryContainer: string;
  inversePrimary: string;

  /* Secondary */
  secondary: string;
  onSecondary: string;
  secondaryContainer: string;
  onSecondaryContainer: string;

  /* Tertiary */
  tertiary: string;
  onTertiary: string;
  tertiaryContainer: string;
  onTertiaryContainer: string;

  /* Error */
  error: string;
  onError: string;
  errorContainer: string;
  onErrorContainer: string;

  /* Outline */
  outline: string;
  outlineVariant: string;

  /* Background */
  background: string;
  onBackground: string;

  /* Canvas */
  canvas: string;
  canvasCode: string;

  /* Panel border */
  panelBorder: string;
  panelBorderHover: string;

  /* Code syntax */
  codeKeyword: string;
  codeNumber: string;
  codeText: string;
}

export interface ThemeTokens {
  colors: ColorTokens;
  id: 'dark' | 'light';
}

/* ─── DARK THEME ─── */
export const darkTheme: ThemeTokens = {
  id: 'dark',
  colors: {
    surface: '#141218',
    surfaceDim: '#141218',
    surfaceBright: '#3b383e',
    surfaceContainerLowest: '#0f0d13',
    surfaceContainerLow: '#1d1b20',
    surfaceContainer: '#211f24',
    surfaceContainerHigh: '#2b292f',
    surfaceContainerHighest: '#36343a',
    surfaceVariant: '#36343a',
    surfaceTint: '#cfbcff',

    onSurface: '#e6e0e9',
    onSurfaceVariant: '#cbc4d2',
    inverseSurface: '#e6e0e9',
    inverseOnSurface: '#322f35',

    primary: '#cfbcff',
    onPrimary: '#381e72',
    primaryContainer: '#6750a4',
    onPrimaryContainer: '#e0d2ff',
    inversePrimary: '#6750a4',

    secondary: '#cdc0e9',
    onSecondary: '#342b4b',
    secondaryContainer: '#4d4465',
    onSecondaryContainer: '#bfb2da',

    tertiary: '#e7c365',
    onTertiary: '#3e2e00',
    tertiaryContainer: '#c9a74d',
    onTertiaryContainer: '#503d00',

    error: '#ffb4ab',
    onError: '#690005',
    errorContainer: '#93000a',
    onErrorContainer: '#ffdad6',

    outline: '#948e9c',
    outlineVariant: '#494551',

    background: '#141218',
    onBackground: '#e6e0e9',

    canvas: '#0a0a0c',
    canvasCode: '#0d0d0f',

    panelBorder: 'rgba(255, 255, 255, 0.08)',
    panelBorderHover: 'rgba(255, 255, 255, 0.16)',

    codeKeyword: '#ff7b72',
    codeNumber: '#79c0ff',
    codeText: '#c9d1d9',
  },
};

/* ─── LIGHT THEME ─── */
export const lightTheme: ThemeTokens = {
  id: 'light',
  colors: {
    surface: '#fef7ff',
    surfaceDim: '#ded8e0',
    surfaceBright: '#fef7ff',
    surfaceContainerLowest: '#ffffff',
    surfaceContainerLow: '#f8f1fa',
    surfaceContainer: '#f2ecf4',
    surfaceContainerHigh: '#ece6ee',
    surfaceContainerHighest: '#e6e0e8',
    surfaceVariant: '#e7e0ec',
    surfaceTint: '#6750a4',

    onSurface: '#1d1b20',
    onSurfaceVariant: '#49454f',
    inverseSurface: '#322f35',
    inverseOnSurface: '#f5eff7',

    primary: '#6750a4',
    onPrimary: '#ffffff',
    primaryContainer: '#eaddff',
    onPrimaryContainer: '#21005d',
    inversePrimary: '#d0bcff',

    secondary: '#625b71',
    onSecondary: '#ffffff',
    secondaryContainer: '#e8def8',
    onSecondaryContainer: '#1d192b',

    tertiary: '#7d5260',
    onTertiary: '#ffffff',
    tertiaryContainer: '#ffd8e4',
    onTertiaryContainer: '#31111d',

    error: '#b3261e',
    onError: '#ffffff',
    errorContainer: '#f9dedc',
    onErrorContainer: '#410e0b',

    outline: '#79747e',
    outlineVariant: '#cac4d0',

    background: '#fef7ff',
    onBackground: '#1d1b20',

    canvas: '#f5f0f7',
    canvasCode: '#f0ebf3',

    panelBorder: 'rgba(0, 0, 0, 0.08)',
    panelBorderHover: 'rgba(0, 0, 0, 0.16)',

    codeKeyword: '#a626a4',
    codeNumber: '#0184bc',
    codeText: '#383a42',
  },
};

/* ─── SPACING ─── */
export const spacing = {
  unit: 4,
  panelGap: 1,
  containerPadding: 8,
  elementGap: 4,
  sidebarWidth: 200,
  controlsWidth: 240,
  toolbarHeight: 40,
  codeHeight: 140,
} as const;

/* ─── TYPOGRAPHY ─── */
export const typography = {
  headerTitle: {
    fontFamily: "'Inter', sans-serif",
    fontSize: '13px',
    fontWeight: 600,
    lineHeight: '1.2',
    letterSpacing: '-0.01em',
  },
  bodyBase: {
    fontFamily: "'Inter', sans-serif",
    fontSize: '11px',
    fontWeight: 400,
    lineHeight: '1.4',
    letterSpacing: '0em',
  },
  labelUppercase: {
    fontFamily: "'Inter', sans-serif",
    fontSize: '10px',
    fontWeight: 600,
    lineHeight: '1',
    letterSpacing: '0.05em',
    textTransform: 'uppercase' as const,
  },
  monoCode: {
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: '11px',
    fontWeight: 400,
    lineHeight: '1.5',
  },
  monoSmall: {
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: '10px',
    fontWeight: 400,
    lineHeight: '1',
  },
} as const;

/* ─── SHAPES ─── */
export const shapes = {
  none: '0px',
  micro: '2px',
  small: '4px',
  medium: '8px',
  large: '12px',
  full: '9999px',
} as const;
