import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';
import Slider from '@mui/material/Slider';

function Copyright() {
  return (
    <Typography
      variant="body2"
      align="center"
      sx={{
        color: 'text.secondary',
      }}
    >
      {'Copyright © '}
      <Link color="inherit" href="https://mui.com/">
        Your Website
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>
  );
}

export default function App() {
  return (
    <Container maxWidth="md">
      <div className="my-8">
        <Typography variant="h4" component="h1" color='primary' sx={{ mb: 2 }}>
          Hello World
        </Typography>
        <Slider
          className="my-4"
          defaultValue={30}
          classes={{ active: 'shadow-none' }}
          slotProps={{ thumb: { className: 'hover:shadow-none' } }}
        />
        <Copyright />
      </div>
    </Container>
  );
}
