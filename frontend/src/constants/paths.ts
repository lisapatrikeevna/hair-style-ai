export const PATH = {
  home: '/',
  editor: '/editor',
} as const;

export type PathType = (typeof PATH)[keyof typeof PATH];