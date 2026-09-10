import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const baseApi = createApi({
  reducerPath: 'baseApi',
  tagTypes: ['HairstyleTask', 'User'],
  baseQuery: fetchBaseQuery({
    baseUrl: API_URL,
    credentials: 'include',
  }),
  endpoints: () => ({}),
});