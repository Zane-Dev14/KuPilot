// API Configuration
// For local development: leave empty string (uses same origin)
// For production (GitHub Pages): set to your Render backend URL
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? ''  // Local: use same origin
  : 'https://kupilot.onrender.com';  // Production: Render backend
