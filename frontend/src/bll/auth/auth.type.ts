export type UserType = {
  id?: number;
  email: string;
  first_name?: string;
  last_name?: string;
  is_guest?: boolean;
  username?: string;
};

export type LoginArgs = {
  password: string;
  username: string;
};

export type responseRegisterType = {
  warning?: string;
};

export type SignUpPayload = {
  email: string;
  password: string;
  username: string;
  password_confirm:string
};