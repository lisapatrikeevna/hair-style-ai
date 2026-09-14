import { createSlice, PayloadAction } from '@reduxjs/toolkit';


export interface InitialStateType {
  isInitialized: boolean;
  isLoading: boolean;
  error: string | null;
  successMessage: string | null;
}

const initialState: InitialStateType = {
  isInitialized: false,
  isLoading: false,
  error: null,
  successMessage: null,
};

const slice = createSlice({
  name: 'app',
  initialState,
  reducers: {
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
  },
});

export const appActions = slice.actions;
export const appReducer = slice.reducer;