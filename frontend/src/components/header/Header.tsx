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
  Chip
} from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import LoginIcon from '@mui/icons-material/Login';
import PersonAddIcon from '@mui/icons-material/PersonAdd';
import LogoutIcon from '@mui/icons-material/Logout';

import { useNavigate } from 'react-router-dom';
import cl from "./Header.module.scss";
import { PATH } from "@/constants/paths.ts";
import logo from '../../assets/logo.svg';

// Temporary auth hook placeholder (will be replaced with actual AuthContext)
const useAuth = () => {
  return { user: null, logout: () => console.log('Logout') };
};

const Header = () => {
  const navigate = useNavigate();
  const { mode, setMode } = useColorScheme();
  const { user, logout } = useAuth();

  // State to handle hamburger menu open/close
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const isMenuOpen = Boolean(anchorEl);

  const handleOpenMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
  };

  const toggleTheme = () => {
    setMode(mode === 'dark' ? 'light' : 'dark');
  };

  const handleLogout = () => {
    handleCloseMenu();
    logout();
  };

  return (
    <AppBar position="sticky" elevation={0} className={cl.header}>
      <Container maxWidth="xl">
        <Toolbar sx={{ justifyContent: 'space-between' }}>

          {/* Brand Logo and Title */}
          <Box
            onClick={() => navigate(PATH.home)}
            sx={{ display: 'flex', alignItems: 'center', gap: 1.5, cursor: 'pointer' }}
          >
            <Box
              component="img"
              src={logo}
              alt="Glam Mirror AI Logo"
              sx={{ width: 36, height: 36, objectFit: 'contain' }}
            />
            <Typography variant="h6" component="div" className={cl.logo} sx={{ fontWeight: 700 }}>
              Glam Mirror <span style={{ opacity: 0.7, fontSize: '0.85em' }}>AI</span>
            </Typography>
          </Box>

          {/* Right Section: Hamburger / User Profile */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {user?.is_guest && (
              <Chip label="Guest" size="small" variant="outlined" color="warning" />
            )}

            <IconButton
              onClick={handleOpenMenu}
              color="inherit"
              aria-label="open user menu"
              aria-controls={isMenuOpen ? 'account-menu' : undefined}
              aria-haspopup="true"
              aria-expanded={isMenuOpen ? 'true' : undefined}
            >
              {user && !user.is_guest ? (
                <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main', fontSize: 14 }}>
                  {user.email?.[0]?.toUpperCase() || 'U'}
                </Avatar>
              ) : (
                <MenuIcon />
              )}
            </IconButton>
          </Box>

          {/* Dropdown Hamburger Menu */}
          <Menu
            anchorEl={anchorEl}
            id="account-menu"
            open={isMenuOpen}
            onClose={handleCloseMenu}
            onClick={handleCloseMenu}
            transformOrigin={{ horizontal: 'right', vertical: 'top' }}
            anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            PaperProps={{
              elevation: 4,
              sx: { minWidth: 200, borderRadius: 2, mt: 1 }
            }}
          >
            {/* User Details Header */}
            <Box sx={{ px: 2, py: 1.5 }}>
              <Typography variant="subtitle2" noWrap>
                {user ? (user.is_guest ? 'Guest Session' : user.email) : 'Welcome!'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {user ? 'Glam Mirror AI' : 'Sign in to your account'}
              </Typography>
            </Box>

            <Divider />

            {/* Theme Toggle Button */}
            <MenuItem onClick={toggleTheme}>
              <ListItemIcon>
                {mode === 'dark' ? <Brightness7Icon fontSize="small" /> : <Brightness4Icon fontSize="small" />}
              </ListItemIcon>
              <ListItemText primary={mode === 'dark' ? 'Light Mode' : 'Dark Mode'} />
            </MenuItem>

            <Divider />

            {/* Authentication Actions */}
            {user && !user.is_guest ? (
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" color="error" />
                </ListItemIcon>
                <ListItemText primary="Logout" color="error" />
              </MenuItem>
            ) : (
              [
                <MenuItem key="login" onClick={() => navigate('/login')}>
                  <ListItemIcon>
                    <LoginIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Login" />
                </MenuItem>,
                <MenuItem key="register" onClick={() => navigate('/register')}>
                  <ListItemIcon>
                    <PersonAddIcon fontSize="small" />
                  </ListItemIcon>
                  <ListItemText primary="Register" />
                </MenuItem>
              ]
            )}
          </Menu>

        </Toolbar>
      </Container>
    </AppBar>
  );
};

export default Header;