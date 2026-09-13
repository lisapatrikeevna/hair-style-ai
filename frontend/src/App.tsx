import { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Box, Snackbar, Alert } from '@mui/material';
import Header from './components/header/Header';
import Footer from './components/footer/Footer';

import cl from './App.module.scss';
import { useGuestLoginMutation, useMeQuery } from "@/bll/auth/auth.serviese.ts";

const App = () => {
  const [notification, setNotification] = useState<{
    message: string | null;
    type: 'error' | 'success' | 'info';
  }>({ message: null, type: 'info' });

  // 1. Query executes automatically on mount and manages caching
  const { error: errorMe, isLoading: isLoadingMe } = useMeQuery();

  // 2. Mutation for initializing guest session
  const [guestLogin, { isLoading: isLoadingGuest }] = useGuestLoginMutation();

  const handleCloseSnackbar = () => {
    setNotification((prev) => ({ ...prev, message: null }));
  };

  // 3. React to 401 Unauthorized automatically
  useEffect(() => {
    if (errorMe && 'status' in errorMe && errorMe.status === 401) {
      guestLogin()
        .unwrap()
        .then((res) => {
          console.log('Guest session created:', res);
        })
        .catch((err) => {
          console.error('Failed to create guest session:', err);
          setNotification({
            message: 'Failed to initialize guest session',
            type: 'error',
          });
        });
    }
  }, [errorMe, guestLogin]);

  const globalLoading = isLoadingMe || isLoadingGuest;

  return (
    <Box className={cl.appShell}>
      <Header />

      <Box component="main" className={cl.mainWrap}>
        {globalLoading && (
          <Box className={cl.loaderOverlay}>
            {/* Overlay Loader */}
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