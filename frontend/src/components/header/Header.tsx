import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  IconButton,
  useColorScheme,
  Box,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Chip,
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import LoginIcon from '@mui/icons-material/Login';
import PersonAddIcon from '@mui/icons-material/PersonAdd';
import LogoutIcon from '@mui/icons-material/Logout';
import { useNavigate } from 'react-router-dom';

import cl from './Header.module.scss';
import { PATH } from '@/constants/paths.ts';
import logo from '../../assets/logo.svg';
import { useLogOutMutation, useMeQuery } from '@/bll/auth/auth.serviese.ts';
import { AuthModal } from '@/components/auth/authModal.tsx';

const Header = () => {
  const navigate = useNavigate();
  const { mode, setMode } = useColorScheme();
  const { data: user } = useMeQuery();
  const [logout] = useLogOutMutation();

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const isMenuOpen = Boolean(anchorEl);

  const [authModalState, setAuthModalState] = useState<{
    open: boolean;
    initialTab: 'login' | 'register';
  }>({ open: false, initialTab: 'login' });

  const handleOpenMenu = (e: React.MouseEvent<HTMLElement>) => setAnchorEl(e.currentTarget);
  const handleCloseMenu = () => setAnchorEl(null);
  const toggleTheme = () => setMode(mode === 'dark' ? 'light' : 'dark');

  const handleLogout = () => {
    handleCloseMenu();
    logout();
  };

  const handleOpenAuth = (tab: 'login' | 'register') => {
    handleCloseMenu();
    setAuthModalState({ open: true, initialTab: tab });
  };

  const handleCloseAuth = () => setAuthModalState((prev) => ({ ...prev, open: false }));

  return (
    <AppBar position="sticky" elevation={0} className={cl.header}>
      <Container maxWidth="xl">
        <Toolbar className={cl.toolbar}>
          <Box onClick={() => navigate(PATH.home)} className={cl.logoBox}>
            <Box component="img" src={logo} alt="Glam Mirror AI Logo" className={cl.logoImg} />
            <Typography variant="h6" component="div" className={cl.logoText}>
              Glam Mirror <span>AI</span>
            </Typography>
          </Box>

          <Box className={cl.controls}>
            {user?.is_guest && <Chip label="Guest" size="small" variant="outlined" color="warning" />}

            <IconButton onClick={handleOpenMenu} color="inherit" aria-label="open user menu">
              {user && !user.is_guest ? (
                <Avatar className={cl.avatar}>{user.email?.[0]?.toUpperCase() || 'U'}</Avatar>
              ) : (
                <MenuIcon />
              )}
            </IconButton>
          </Box>

          <Menu anchorEl={anchorEl} id="account-menu" open={isMenuOpen} onClose={handleCloseMenu} onClick={handleCloseMenu}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            slotProps={{ paper: { className: cl.menuPaper } }}
          >
            <Box className={cl.userInfoBox}>
              <Typography variant="subtitle2" noWrap>
                {user ? (user.is_guest ? 'Guest Session' : user.email) : 'Welcome!'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {user ? 'Glam Mirror AI' : 'Please sign in'}
              </Typography>
            </Box>

            <Divider />

            <MenuItem onClick={toggleTheme}>
              <ListItemIcon>
                {mode === 'dark' ? <Brightness7Icon fontSize="small" /> : <Brightness4Icon fontSize="small" />}
              </ListItemIcon>
              <ListItemText primary={mode === 'dark' ? 'Light mode' : 'Dark mode'} />
            </MenuItem>

            <Divider />

            {user && !user.is_guest ? (
              <MenuItem onClick={handleLogout}>
                <ListItemIcon><LogoutIcon fontSize="small" color="error" /></ListItemIcon>
                <ListItemText primary="Log out" />
              </MenuItem>
            ) : (
              [
                <MenuItem key="login" onClick={() => handleOpenAuth('login')}>
                  <ListItemIcon><LoginIcon fontSize="small" /></ListItemIcon>
                  <ListItemText primary="Sign In" />
                </MenuItem>,
                <MenuItem key="register" onClick={() => handleOpenAuth('register')}>
                  <ListItemIcon><PersonAddIcon fontSize="small" /></ListItemIcon>
                  <ListItemText primary="Sign Up" />
                </MenuItem>,
              ]
            )}
          </Menu>
        </Toolbar>
      </Container>

      <AuthModal open={authModalState.open} onClose={handleCloseAuth} initialTab={authModalState.initialTab} />
    </AppBar>
  );
};

export default Header;