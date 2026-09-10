import { createBrowserRouter, Navigate, RouterProvider } from 'react-router-dom';
import App from './App';
import HomePage from '@/pages/home/HomePage';
import EditorPage from '@/pages/editor/EditorPage';
import {PATH} from "./constants/paths.ts";


const router = createBrowserRouter([
  {
    element: <App />,
    children: [
      { path: PATH.home, element: <HomePage /> },
      { path: PATH.editor, element: <EditorPage /> },
      { path: '*', element: <Navigate to={PATH.home} replace /> },
    ],
  },
]);

export const Router = () => {
  return <RouterProvider router={router} />;
};