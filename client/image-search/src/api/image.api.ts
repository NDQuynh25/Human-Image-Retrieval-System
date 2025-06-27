import axios from '../config/axios.config';

export const searchImageAPI = async (formData: FormData) => {
  const response = await axios.post('/api/v1/search', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};