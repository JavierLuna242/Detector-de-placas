// frontend/src/services/api.js

const DEFAULT_API_URL =
  (typeof window !== 'undefined' && localStorage.getItem('plate_api_url')) ||
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) ||
  'http://localhost:8080';

export const getApiUrl = () => {
  return localStorage.getItem('plate_api_url') || DEFAULT_API_URL;
};

export const setApiUrl = (url) => {
  let formattedUrl = url.trim().replace(/\/+$/, '');
  if (!formattedUrl.startsWith('http://') && !formattedUrl.startsWith('https://')) {
    formattedUrl = 'http://' + formattedUrl;
  }
  localStorage.setItem('plate_api_url', formattedUrl);
  return formattedUrl;
};

export const checkApiHealth = async () => {
  const baseUrl = getApiUrl();
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(`${baseUrl}/`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      return { online: true, message: data.message || 'Servidor Activo' };
    }
    return { online: false, message: `HTTP Error ${response.status}` };
  } catch (error) {
    return { online: false, message: 'No se puede conectar al servidor FastAPI' };
  }
};

export const predictPlate = async (imageInput) => {
  const baseUrl = getApiUrl();
  
  try {
    let response;
    
    if (typeof imageInput === 'string' && imageInput.startsWith('data:image')) {
      // Petición JSON con base64
      response = await fetch(`${baseUrl}/predict_json/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: imageInput,
        }),
      });
    } else if (imageInput instanceof Blob || imageInput instanceof File) {
      // Petición multipart/form-data
      const formData = new FormData();
      formData.append('file', imageInput);
      response = await fetch(`${baseUrl}/predict/`, {
        method: 'POST',
        body: formData,
      });
    } else {
      // Petición Form Data con base64
      const formData = new FormData();
      formData.append('image_base64', imageInput);
      response = await fetch(`${baseUrl}/predict/`, {
        method: 'POST',
        body: formData,
      });
    }

    if (!response.ok) {
      throw new Error(`Error en el servidor: HTTP ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error al realizar predicción:', error);
    return {
      success: false,
      error: error.message || 'Error de conexión con el backend',
      placas: [],
      num_placas: 0,
      image: null,
    };
  }
};
