import {StrictMode} from 'react';
import './index.scss'; // Updated SCSS import
import ReactDOM from 'react-dom/client';
import {ThemeProvider} from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import InitColorSchemeScript from '@mui/material/InitColorSchemeScript';
import {theme} from './theme/theme.ts';
import {Router} from './router';
import {Provider} from "react-redux";
import {store} from "@/bll/store.ts";

ReactDOM.createRoot(document.getElementById('root')!).render(<StrictMode>
  <Provider store={store}>
    {/* Injects the theme mode script synchronously before rendering */}
    <InitColorSchemeScript defaultMode="dark"/>
    <ThemeProvider theme={theme} defaultMode="dark">
      <CssBaseline/>
      <Router/>
    </ThemeProvider>
  </Provider>
</StrictMode>);