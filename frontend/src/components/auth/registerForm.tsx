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
} from '@mui/material';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import { useSignUpMutation } from '@/bll/auth/auth.serviese';
import cl from './auth.module.scss';
import {handleError} from "@/helpers/handleError.ts";
import {useDispatch} from "react-redux";

const registerSchema = z
  .object({
    email: z.string().email('Invalid email address').min(1, 'Enter email'),
    identifier: z.string().min(1, 'Enter nickname'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters')
      .refine((val) => !/^\d+$/.test(val), 'Password cannot be entirely numeric'),
    passwordConfirmation: z.string().min(1, 'Confirm your password'),
  })
  .superRefine((data, ctx) => {
    if (data.password !== data.passwordConfirmation) {
      ctx.addIssue({
        message: 'Passwords do not match',
        code: z.ZodIssueCode.custom,
        path: ['passwordConfirmation'],
      });
    }
  });

type RegisterFormData = z.infer<typeof registerSchema>;

interface RegisterFormProps {
  onSuccess: () => void;
  onSwitchToLogin: () => void;
}

export const RegisterForm = ({ onSuccess, onSwitchToLogin }: RegisterFormProps) => {
  const dispatch = useDispatch();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [signUp, { isLoading }] = useSignUpMutation();

  const {register, handleSubmit, formState: { errors },} = useForm<RegisterFormData>({
    mode: 'onSubmit',
    resolver: zodResolver(registerSchema),
    defaultValues: { email: '', identifier: '', password: '', passwordConfirmation: '' },
  });

  const onSubmit = (data: RegisterFormData) => {
    signUp({ email: data.email, password: data.password, username: data.identifier ,password_confirm: data.passwordConfirmation })
      .unwrap().then(() => onSuccess())
      .catch((err) => {
        // console.log('Registration error:', err);
        handleError(err?.data?.detail || err ,'Registration failed',dispatch);
      });
  };

  return (
    <Box component="form" onSubmit={handleSubmit(onSubmit)} noValidate className={cl.form}>

      <TextField margin="normal" fullWidth label="Email" type="email" autoFocus {...register('email')}
        error={!!errors.email} helperText={errors.email?.message}/>

      <TextField margin="normal" fullWidth label="Nickname"{...register('identifier')}
        error={!!errors.identifier} helperText={errors.identifier?.message}/>

      <TextField margin="normal" fullWidth label="Password" type={showPassword ? 'text' : 'password'}
        {...register('password')} error={!!errors.password} helperText={errors.password?.message}
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

      <TextField margin="normal" fullWidth label="Confirm Password" type={showConfirmPassword ? 'text' : 'password'}
        {...register('passwordConfirmation')} error={!!errors.passwordConfirmation} helperText={errors.passwordConfirmation?.message}
        slotProps={{
          input: {
            endAdornment: (
              <InputAdornment position="end">
                <IconButton onClick={() => setShowConfirmPassword((prev) => !prev)} edge="end">
                  {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          },
        }}
      />

      <Button type="submit" fullWidth variant="contained" disabled={isLoading} className={cl.submitButton}>
        {isLoading ? 'Signing up...' : 'Sign Up'}
      </Button>

      <Box className={cl.switchBox}>
        <Typography variant="body2" color="text.secondary">
          Already have an account?{' '}
          <Button variant="text" onClick={onSwitchToLogin} className={cl.switchButton}>
            Sign In
          </Button>
        </Typography>
      </Box>
    </Box>
  );
};