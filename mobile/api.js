import Constants from 'expo-constants';

export const getApiUrl = () => {
  const extra = Constants.expoConfig?.extra || {};
  const configured = extra.apiUrl || process.env.EXPO_PUBLIC_API_URL || 'http://52.3.161.151:8080';
  return configured.replace(/\/+$/, '');
};

export const detectPlate = async (imageUri, fileLike = null) => {
  const apiUrl = getApiUrl();
  const formData = new FormData();

  if (typeof File !== 'undefined' && fileLike instanceof File) {
    formData.append('file', fileLike);
  } else {
    formData.append('file', {
      uri: imageUri,
      name: 'plate.jpg',
      type: 'image/jpeg',
      ...(fileLike || {}),
    });
  }

  const response = await fetch(`${apiUrl}/predict/`, {
    method: 'POST',
    body: formData,
    headers: {
      Accept: 'application/json',
    },
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.error || 'Error al consultar el backend');
  }

  return data;
};
