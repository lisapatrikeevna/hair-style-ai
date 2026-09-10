import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface UserType {
  id: string;
  email?: string;
  isGuest: boolean;
}

export interface InitialStateType {
  user: UserType | null;
  isInitialized: boolean;
  isLoading: boolean;
  error: string | null;
  successMessage: string | null;
}

const initialState: InitialStateType = {
  user: null,
  isInitialized: false,
  isLoading: false,
  error: null,
  successMessage: null,
};

const slice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setUser: (state, action: PayloadAction<UserType | null>) => {
      state.user = action.payload;
    },
    setIsInitialized: (state, action: PayloadAction<boolean>) => {
      state.isInitialized = action.payload;
    },
    setIsLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setSuccessMessage: (state, action: PayloadAction<string | null>) => {
      state.successMessage = action.payload;
    },
    logout: (state) => {
      state.user = null;
    },
  },
});

export const appActions = slice.actions;
export const appReducer = slice.reducer;