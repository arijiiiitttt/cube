import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "react-hot-toast";
import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
        <App />
        <Toaster
          position="top-right"
          toastOptions={{
            className: "text-sm font-medium",
            style: {
              borderRadius: "10px",
              background: "#0f172a",
              color: "#f8fafc",
            },
          }}
        />
    </BrowserRouter>
  </React.StrictMode>
);
