import { useState } from 'react';
import { Dialog, DialogTitle, DialogContent, Tabs, Tab, Box, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import cl from './auth.module.scss';
import { RegisterForm } from '@/components/auth/registerForm';
import { LoginForm } from './loginForm';

interface AuthModalProps {
  open: boolean;
  onClose: () => void;
  initialTab?: 'login' | 'register';
}

export const AuthModal = ({ open, onClose, initialTab = 'login' }: AuthModalProps) => {
  const [activeTab, setActiveTab] = useState<'login' | 'register'>(initialTab);

  const handleTabChange = (_: React.SyntheticEvent, newValue: 'login' | 'register') => {
    setActiveTab(newValue);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth className={cl.authModal}>
      <DialogTitle className={cl.modalTitle}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab label="Sign In" value="login" />
          <Tab label="Sign Up" value="register" />
        </Tabs>
        <IconButton onClick={onClose} size="small" className={cl.closeButton}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent dividers className={cl.modalContent}>
        <Box className={cl.formContainer}>
          {activeTab === 'login' ? (
            <LoginForm onSuccess={onClose} onSwitchToRegister={() => setActiveTab('register')} />
          ) : (
            <RegisterForm onSuccess={onClose} onSwitchToLogin={() => setActiveTab('login')} />
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
};