import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Box, Snackbar, Alert } from '@mui/material';
import Header from './components/header/Header';
import Footer from './components/footer/Footer';

import cl from './App.module.scss';

const App = () => {
  const [isLoading] = useState<boolean>(false);
  const [notification, setNotification] = useState<{
    message: string | null;
    type: 'error' | 'success' | 'info';
  }>({ message: null, type: 'info' });

  const handleCloseSnackbar = () => {
    setNotification((prev) => ({ ...prev, message: null }));
  };

  return (
    <Box className={cl.appShell}>
      <Header />

      <Box component="main" className={cl.mainWrap}>
        {/* Overlay loader */}
        {isLoading && (
          <Box className={cl.loaderOverlay}>
            {/* Prelouder component */}
          </Box>
        )}

        <Box className={cl.contentWrap}>
          <Outlet />
        </Box>

        <Snackbar
          open={!!notification.message}
          autoHideDuration={6000}
          onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert
            onClose={handleCloseSnackbar}
            severity={notification.type}
            variant="filled"
            className={cl.snackbarAlert}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      </Box>

      <Footer />
    </Box>
  );
};

export default App;