import { baseApi } from "../base-api";
import { LoginArgs, responseRegisterType, SignUpPayload, UserType } from './auth.type';

const authService = baseApi.injectEndpoints({
  endpoints: (builder) => ({
    // Fetches currently authenticated user or active guest profile
    me: builder.query<UserType, void>({
      query: () => ({
        url: '/auth/me/',
        method: 'GET',
      }),
      extraOptions: { maxRetries: 0 },
      providesTags: ['Me'],
    }),

    // Authenticates user and sets httpOnly cookies from Django backend
    login: builder.mutation<responseRegisterType, LoginArgs>({
      query: (args) => ({
        url: '/auth/login/',
        method: 'POST',
        body: args, // RTK Query automatically serializes objects to JSON
      }),
      invalidatesTags: ['Me'],
    }),

    // Creates temporary guest session with httpOnly cookie
    guestLogin: builder.mutation<responseRegisterType, void>({
      query: () => ({
        url: '/auth/guest/',
        method: 'POST',
      }),
      invalidatesTags: ['Me'],
    }),

    // Registers a new regular user
    signUp: builder.mutation<responseRegisterType, SignUpPayload>({
      query: (args) => ({
        url: '/auth/register/',
        method: 'POST',
        body: args,
      }),
      invalidatesTags: ['Me'],
    }),

    // Logs out user, blacklists token and clears backend cookies
    logOut: builder.mutation<void, void>({
      query: () => ({
        url: '/auth/logout/',
        method: 'POST',
      }),
      invalidatesTags: ['Me'],
    }),
  }),
});

export const {
  useMeQuery,
  useLoginMutation,
  useGuestLoginMutation,
  useSignUpMutation,
  useLogOutMutation,
} = authService;