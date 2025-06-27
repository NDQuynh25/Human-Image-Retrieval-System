// src/utils/axiosInstance.ts
import axios from 'axios';


const axiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000',
  timeout: 30000, // Timeout sau 10s
  headers: {
    'Content-Type': 'application/json',
    // 'Authorization': `Bearer ${token}` // nếu cần
  },
});

// Optional: Thêm interceptors để tự động xử lý request/response
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  },
  (error) => Promise.reject(error)
);

axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // handle logout hoặc refresh token
      console.error('Unauthorized, redirect to login...');
    }
    return Promise.reject(error);
  }
);

export default axiosInstance;
