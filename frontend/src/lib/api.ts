import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  timeout: 30000,
});

// ── Upload & Dataset ────────────────────────────────────────────────────────
export const uploadFiles = (form: FormData) =>
  api.post("/upload", form, { headers: { "Content-Type": "multipart/form-data" } });

export const processDataset = (id: string) => api.post(`/dataset/${id}/process`);
export const getDatasetStatus = (id: string) => api.get(`/dataset/${id}/status`);
export const listDatasets = () => api.get("/dataset");

// ── Training ────────────────────────────────────────────────────────────────
export const startTraining = (cfg: object) => api.post("/training/start", cfg);
export const estimateTraining = (req: object) => api.post("/training/estimate", req);
export const stopTraining = (jobId: string) => api.post(`/training/${jobId}/stop`);
export const getTrainingStatus = (jobId: string) => api.get(`/training/${jobId}/status`);
export const getTrainingLogs = (jobId: string) => api.get(`/training/${jobId}/logs`);
export const listJobs = () => api.get("/training");

// ── Evaluation ──────────────────────────────────────────────────────────────
export const runEvaluation = (modelId: string) => api.post(`/eval/${modelId}/run`);
export const getEvalResults = (modelId: string) => api.get(`/eval/${modelId}/results`);

// ── Models & Deploy ─────────────────────────────────────────────────────────
export const listModels = () => api.get("/models");
export const getModel = (id: string) => api.get(`/models/${id}`);
export const generateText = (modelId: string, data: object) =>
  api.post(`/inference/${modelId}/generate`, data);
export const deployModel = (modelId: string) => api.post(`/deploy/${modelId}`);
export const getDeployInfo = (modelId: string) => api.get(`/deploy/${modelId}`);

export default api;
