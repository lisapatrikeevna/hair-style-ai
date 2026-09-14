import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import {API_URL} from "@/config.ts";

const BASE_URL = `${API_URL}/api/v1/`;

export const baseApi = createApi({
  reducerPath: 'baseApi',
  tagTypes: ['HairstyleTask', 'User','Me'],
  baseQuery: fetchBaseQuery({
    baseUrl: BASE_URL,
    credentials: 'include',
  }),
  endpoints: () => ({}),
});