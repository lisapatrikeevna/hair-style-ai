import { createTheme } from '@mui/material/styles';

export const BRAND_COLORS = {
  light: {
    primary: '#0086C2',
    secondary: '#FF6B29',
    bgDefault: '#F6F7FB',
    bgPaper: '#FFFFFF',
    textPrimary: '#1E222E',
    textSecondary: '#414040',
  },
  dark: {
    primary: '#0086C2',
    secondary: '#BED1F2',
    bgDefault: '#00091B',
    bgPaper: '#03122E',
    textPrimary: '#E5E7EB',
    textSecondary: '#9CA3AF',
  },
};

export const theme = createTheme({
  cssVariables: {
    colorSchemeSelector: 'class', // Adds .mui-mode-light / .mui-mode-dark to html element
  },
  colorSchemes: {
    light: {
      palette: {
        mode: 'light',
        primary: { main: BRAND_COLORS.light.primary },
        secondary: { main: BRAND_COLORS.light.secondary },
        background: {
          default: BRAND_COLORS.light.bgDefault,
          paper: BRAND_COLORS.light.bgPaper,
        },
        text: {
          primary: BRAND_COLORS.light.textPrimary,
          secondary: BRAND_COLORS.light.textSecondary,
        },
      },
    },
    dark: {
      palette: {
        mode: 'dark',
        primary: { main: BRAND_COLORS.dark.primary },
        secondary: { main: BRAND_COLORS.dark.secondary },
        background: {
          default: BRAND_COLORS.dark.bgDefault,
          paper: BRAND_COLORS.dark.bgPaper,
        },
        text: {
          primary: BRAND_COLORS.dark.textPrimary,
          secondary: BRAND_COLORS.dark.textSecondary,
        },
      },
    },
  },
});