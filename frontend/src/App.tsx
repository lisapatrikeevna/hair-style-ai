import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { Box, Snackbar, Alert, CircularProgress } from '@mui/material';
import Header from './components/header/Header';
import Footer from './components/footer/Footer';

import cl from './App.module.scss';
import { useGuestLoginMutation, useMeQuery } from '@/bll/auth/auth.serviese.ts';
import { appActions } from '@/bll/app.slice.ts';
import {useAppDispatch, useAppSelector} from "@/bll/store.ts";
import {handleError} from "@/helpers/handleError.ts";

const App = () => {
  const dispatch = useAppDispatch();
  const { isInitialized, error, successMessage } = useAppSelector((state) => state.app);

  console.log('error', error);
  console.log('successMessage', successMessage);

  const { error: errorMe, isLoading: isLoadingMe, isSuccess, isError } = useMeQuery();
  const [guestLogin, { isLoading: isLoadingGuest }] = useGuestLoginMutation();

  const handleCloseSnackbar = () => {
    dispatch(appActions.setError(null));
    dispatch(appActions.setSuccessMessage(null));
  };

  useEffect(() => {
    if (isSuccess) dispatch(appActions.setIsInitialized(true));

    if (isError && errorMe && 'status' in errorMe && errorMe.status === 401) {
      guestLogin()
        .unwrap()
        .then(() => dispatch(appActions.setIsInitialized(true)))
        .catch((err) => handleError(err, 'Failed to initialize guest session', dispatch));
    }
  }, [errorMe, isSuccess, isError, guestLogin, dispatch]);

  const globalLoading = !isInitialized || isLoadingMe || isLoadingGuest;

  return (
    <Box className={cl.appShell}>
      <Header />

      <Box component="main" className={cl.mainWrap}>
        {globalLoading ? (
          <Box className={cl.loaderOverlay}><CircularProgress /></Box>
        ) : (
          <Box className={cl.contentWrap}><Outlet /></Box>
        )}

        <Snackbar open={!!error || !!successMessage} autoHideDuration={6000} onClose={handleCloseSnackbar}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
          <Alert onClose={handleCloseSnackbar} severity={error ? 'error' : 'success'} variant="filled" className={cl.snackbarAlert}>
            {error || successMessage}
          </Alert>
        </Snackbar>
      </Box>

      <Footer />
    </Box>
  );
};

export default App;