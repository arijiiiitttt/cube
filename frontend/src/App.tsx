import { Routes, Route, Navigate } from "react-router-dom";
import DashboardLayout from "./components/DashboardLayout";
import UploadPage from "./pages/Upload";
import DatasetPage from "./pages/Dataset";
import ModelPage from "./pages/Model";
import TrainingPage from "./pages/Training";
import EvaluationPage from "./pages/Evaluation";
import DeployPage from "./pages/Deploy";

export default function App() {
  return (
    <Routes>
      <Route
        path="/dashboard"
        element={<DashboardLayout />}
      >
        <Route index element={<Navigate to="upload" replace />} />
        <Route path="upload" element={<UploadPage />} />
        <Route path="dataset" element={<DatasetPage />} />
        <Route path="model" element={<ModelPage />} />
        <Route path="training" element={<TrainingPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="deploy" element={<DeployPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/dashboard/upload" replace />} />
    </Routes>
  );
}
