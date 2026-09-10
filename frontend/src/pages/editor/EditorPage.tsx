import {useState, ChangeEvent, DragEvent} from 'react';
import {Container, Typography, Box, Button, Card, CardContent, Stack, Grid} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PhotoCameraIcon from '@mui/icons-material/PhotoCamera';
import DeleteIcon from '@mui/icons-material/Delete';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import {useNavigate} from 'react-router-dom';
import {PATH} from '@/constants/paths';

import cl from './EditorPage.module.scss';

const EditorPage = () => {
  const navigate = useNavigate();

  const [mainPhoto, setMainPhoto] = useState<string | null>(null);
  const [ponytailPhoto, setPonytailPhoto] = useState<string | null>(null);
  const [dragActiveSlot, setDragActiveSlot] = useState<'main' | 'ponytail' | null>(null);

  const handleFileSelect = (file: File | undefined, setImage: (val: string | null) => void) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>, setImage: (val: string | null) => void) => {
    const file = e.target.files?.[0];
    handleFileSelect(file, setImage);
  };

  const handleDragOver = (e: DragEvent<HTMLLabelElement>, slot: 'main' | 'ponytail') => {
    e.preventDefault();
    e.stopPropagation();
    setDragActiveSlot(slot);
  };

  const handleDragLeave = (e: DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActiveSlot(null);
  };

  const handleDrop = (e: DragEvent<HTMLLabelElement>, setImage: (val: string | null) => void) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActiveSlot(null);
    const file = e.dataTransfer.files?.[0];
    handleFileSelect(file, setImage);
  };

  return (<Container maxWidth="lg" className={cl.container}>
    <Button startIcon={<ArrowBackIcon/>} onClick={() => navigate(PATH.home)} sx={{mb: 3}}>
      Back to Home
    </Button>

    <Typography variant="h4" className={cl.title}>
      Upload Photos for AI Styling
    </Typography>
    <Typography variant="body1" className={cl.subtitle}>
      Provide your main photo to detect hair structure, and an optional ponytail photo for clean face contouring.
    </Typography>

    <Grid container spacing={4}>
      <Grid size={{xs: 12, md: 6}}>
        <Typography variant="h6" sx={{mb: 1, textAlign: 'center'}}>
          1. Main Photo (Required)
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{mb: 2, textAlign: 'center'}}>
          Upload a photo where your hair structure is clearly visible (ideally not covering your face).
        </Typography>

        {!mainPhoto ? (<Card
          className={`${cl.dropZone} ${dragActiveSlot === 'main' ? cl.dragActive : ''}`}
          component="label"
          onDragOver={(e) => handleDragOver(e, 'main')}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, setMainPhoto)}
        >
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleInputChange(e, setMainPhoto)}
          />
          <CardContent>
            <CloudUploadIcon sx={{fontSize: 48, color: 'primary.main', mb: 1}}/>
            <Typography variant="h6" gutterBottom>
              Drag & Drop or Click to Upload
            </Typography>
            <Stack direction="row" spacing={1} sx={{justifyContent: "center", mt: 2}}>
              <Button
                variant="outlined"
                size="small"
                startIcon={<PhotoCameraIcon/>}
                onClick={(e) => {
                  e.preventDefault();
                  // Camera capture logic stub
                }}
              >
                Take Photo
              </Button>
            </Stack>
          </CardContent>
        </Card>) : (<Box sx={{textAlign: 'center'}}>
          <img src={mainPhoto} alt="Main photo preview" className={cl.previewImage}/>
          <Button
            variant="outlined"
            color="error"
            startIcon={<DeleteIcon/>}
            onClick={() => setMainPhoto(null)}
          >
            Remove
          </Button>
        </Box>)}
      </Grid>

      <Grid size={{xs: 12, md: 6}}>
        <Typography variant="h6" sx={{mb: 1, textAlign: 'center'}}>
          2. Ponytail Photo (Optional)
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{mb: 2, textAlign: 'center'}}>
          Upload a photo with hair tied back or pulled away from the face for precise forehead detection.
        </Typography>

        {!ponytailPhoto ? (<Card
          className={`${cl.dropZone} ${dragActiveSlot === 'ponytail' ? cl.dragActive : ''}`}
          component="label"
          onDragOver={(e) => handleDragOver(e, 'ponytail')}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, setPonytailPhoto)}
        >
          <input
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleInputChange(e, setPonytailPhoto)}
          />
          <CardContent>
            <CloudUploadIcon sx={{fontSize: 48, color: 'secondary.main', mb: 1}}/>
            <Typography variant="h6" gutterBottom>
              Drag & Drop or Click to Upload
            </Typography>
            <Stack direction="row" spacing={1} sx={{justifyContent: "center", mt: 2}}>
              <Button
                variant="outlined"
                size="small"
                startIcon={<PhotoCameraIcon/>}
                onClick={(e) => {
                  e.preventDefault();
                  // Camera capture logic stub
                }}
              >
                Take Photo
              </Button>
            </Stack>
          </CardContent>
        </Card>) : (<Box sx={{textAlign: 'center'}}>
          <img src={ponytailPhoto} alt="Ponytail photo preview" className={cl.previewImage}/>
          <Button
            variant="outlined"
            color="error"
            startIcon={<DeleteIcon/>}
            onClick={() => setPonytailPhoto(null)}
          >
            Remove
          </Button>
        </Box>)}
      </Grid>
    </Grid>

    <Box className={cl.actions}>
      <Button
        variant="contained"
        size="large"
        disabled={!mainPhoto}
        onClick={() => {
          // Next step action
        }}
      >
        Generate Hairstyles
      </Button>
    </Box>
  </Container>);
};

export default EditorPage;