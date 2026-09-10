import {Container, Typography, Card, CardContent, CardActions, Button, Chip, Box, Grid} from '@mui/material';
import {useNavigate} from 'react-router-dom';
import ContentCutIcon from '@mui/icons-material/ContentCut';
import FaceRetouchingNaturalIcon from '@mui/icons-material/FaceRetouchingNatural';
import Face3Icon from '@mui/icons-material/Face3';
import {PATH} from '@/constants/paths';

import cl from './HomePage.module.scss';

const HomePage = () => {
  const navigate = useNavigate();

  const services = [{
    id: 'hairstyle',
    title: 'Hairstyle & Color',
    description: 'Try on different hairstyles and experiment with hair colors using AI.',
    icon: <ContentCutIcon fontSize="large" color="primary"/>,
    active: true,
    actionText: 'Start Styling',
  }, {
    id: 'makeup',
    title: 'AI Makeup',
    description: 'Apply realistic makeup presets and color palettes directly on your photo.',
    icon: <FaceRetouchingNaturalIcon fontSize="large" color="disabled"/>,
    active: false,
    actionText: 'Coming Soon',
  }, {
    id: 'avatar',
    title: '3D Avatar Generator',
    description: 'Generate a realistic 3D avatar from a single facial photograph.',
    icon: <Face3Icon fontSize="large" color="disabled"/>,
    active: false,
    actionText: 'Coming Soon',
  },];

  return (<Container maxWidth="lg" className={cl.container}>
      <Typography variant="h3" className={cl.heroTitle}>
        AI Beauty Studio
      </Typography>
      <Typography variant="h6" className={cl.heroSubtitle}>
        Select a service below to transform your appearance using machine learning models.
      </Typography>

      <Grid container spacing={4}>
        {services.map((service) => (<Grid size={{xs: 12, md: 4}} key={service.id}>
            <Card
              className={`${cl.card} ${!service.active ? cl.disabledCard : ''}`}
              sx={{position: 'relative'}}
            >
              {!service.active && (<Chip label="Soon" size="small" color="default" className={cl.badge}/>)}
              <CardContent>
                <Box sx={{mb: 2}}>{service.icon}</Box>
                <Typography variant="h5" component="div" gutterBottom>
                  {service.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {service.description}
                </Typography>
              </CardContent>
              <CardActions sx={{p: 2, pt: 0}}>
                <Button variant={service.active ? 'contained' : 'outlined'} fullWidth
                  disabled={!service.active} onClick={() => navigate(PATH.editor)}
                >
                  {service.actionText}
                </Button>
              </CardActions>
            </Card>
          </Grid>))}
      </Grid>
    </Container>);
};

export default HomePage;