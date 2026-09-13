import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import {
  Box,
  Button,
  TextField,
  Typography,
  InputAdornment,
  IconButton,
  Alert,
} from '@mui/material';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import { useLoginMutation } from '@/bll/auth/auth.serviese';
import cl from './auth.module.scss';

const loginSchema = z.object({
  username: z.string().min(1, 'Enter username or email'),
  password: z.string().min(1, 'Enter password'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export interface LoginFormProps {
  onSuccess: () => void;
  onSwitchToRegister: () => void;
}

export const LoginForm = ({ onSuccess, onSwitchToRegister }: LoginFormProps) => {
  const [showPassword, setShowPassword] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [logIn, { isLoading }] = useLoginMutation();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginFormData>({
    mode: 'onSubmit',
    resolver: zodResolver(loginSchema),
    defaultValues: { username: '', password: '' },
  });

  const onSubmit = (data: LoginFormData) => {
    setErrorMessage(null);
    logIn(data)
      .unwrap()
      .then(() => onSuccess())
      .catch((err:any) => {
        console.error('Login error:', err);
        setErrorMessage(err?.data?.detail || err?.data?.message || 'Login failed');
      });
  };

  return (
    <Box component="form" onSubmit={handleSubmit(onSubmit)} noValidate className={cl.form}>
      {errorMessage && <Alert severity="error" className={cl.alert}>{errorMessage}</Alert>}

      <TextField
        margin="normal"
        fullWidth
        label="Username / Email"
        autoFocus
        {...register('username')}
        error={!!errors.username}
        helperText={errors.username?.message}
      />

      <TextField
        margin="normal"
        fullWidth
        label="Password"
        type={showPassword ? 'text' : 'password'}
        {...register('password')}
        error={!!errors.password}
        helperText={errors.password?.message}
        slotProps={{
          input: {
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={() => setShowPassword((prev) => !prev)} edge="end">
                  {showPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
      />

      <Button type="submit" fullWidth variant="contained" disabled={isLoading} className={cl.submitButton}>
        {isLoading ? 'Signing in...' : 'Sign In'}
      </Button>

      <Box className={cl.switchBox}>
        <Typography variant="body2" color="text.secondary">
          Don't have an account?{' '}
          <Button variant="text" onClick={onSwitchToRegister} className={cl.switchButton}>
            Sign Up
          </Button>
        </Typography>
      </Box>
    </Box>
  );
};