import { Box, Container, Typography } from '@mui/material';

import cl from './Footer.module.scss';

const Footer = () => {
  return (
    <Box component="footer" className={cl.footer}>
      <Container maxWidth="xl">
        <Typography variant="body2" className={cl.text}>
          © {new Date().getFullYear()} HairStyle AI. All rights reserved.
        </Typography>
      </Container>
    </Box>
  );
};

export default Footer;