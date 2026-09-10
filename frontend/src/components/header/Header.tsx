import {AppBar, Toolbar, Typography, Container, IconButton, useColorScheme} from '@mui/material';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import {useNavigate} from 'react-router-dom';
import cl from "./Header.module.scss"
import {PATH} from "@/constants/paths.ts";

const Header = () => {
  const navigate = useNavigate();
  const { mode, setMode } = useColorScheme();

  const toggleTheme = () => {
    setMode(mode === 'dark' ? 'light' : 'dark');
  };

  return (<AppBar position="sticky" elevation={0} className={cl.header}>
      <Container maxWidth="xl">
        <Toolbar sx={{justifyContent: 'space-between'}}>
          <Typography variant="h6" className={cl.logo} onClick={() => navigate(PATH.home)}>
            HairStyle AI
          </Typography>
          <IconButton color="inherit" onClick={toggleTheme} aria-label="toggle theme">
            {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
          </IconButton>
        </Toolbar>
      </Container>
    </AppBar>);
};

export default Header;