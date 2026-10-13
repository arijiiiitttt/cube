import { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import {
  Upload,
  FileText,
  FileJson,
  File,
  X,
  CheckCircle,
  AlertCircle,
  ArrowRight,
} from "lucide-react";
import toast from "react-hot-toast";
import { uploadFiles } from "../lib/api";
import { cn, fmtSize } from "../lib/utils";

type FileStatus = "ready" | "uploading" | "done" | "error";

export default function UploadPage() {
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [statuses, setStatuses] = useState<Record<string, FileStatus>>({});
  const [uploading, setUploading] = useState(false);
  const [datasetId, setDatasetId] = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    setFiles((prev) => {
      const names = new Set(prev.map((f) => f.name));
      const unique = accepted.filter((f) => !names.has(f.name));
      return [...prev, ...unique];
    });
    setStatuses((prev) => {
      const next = { ...prev };
      accepted.forEach((f) => {
        if (!next[f.name]) next[f.name] = "ready";
      });
      return next;
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/plain": [".txt"],
      "text/csv": [".csv"],
      "application/json": [".json", ".jsonl"],
    },
    multiple: true,
  });

  const remove = (name: string) => {
    setFiles((f) => f.filter((x) => x.name !== name));
    setStatuses((s) => {
      const n = { ...s };
      delete n[name];
      return n;
    });
  };

  const getIcon = (name: string) => {
    if (name.endsWith(".json") || name.endsWith(".jsonl")) return FileJson;
    if (name.endsWith(".csv") || name.endsWith(".txt")) return FileText;
    return File;
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    setStatuses((s) => {
      const n = { ...s };
      files.forEach((f) => (n[f.name] = "uploading"));
      return n;
    });
    try {
      const res = await uploadFiles(form);
      const id = res.data.dataset_id || res.data.id;
      setDatasetId(id);
      setStatuses((s) => {
        const n = { ...s };
        files.forEach((f) => (n[f.name] = "done"));
        return n;
      });
      toast.success("Upload complete");
    } catch (err: any) {
      setStatuses((s) => {
        const n = { ...s };
        files.forEach((f) => (n[f.name] = "error"));
        return n;
      });
      toast.error(err.response?.data?.detail || "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const allDone = files.length > 0 && files.every((f) => statuses[f.name] === "done");

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <h1 className="page-title">Upload Dataset</h1>
        <p className="page-desc">
          Drop CSV, JSON, JSONL or plain text files. We'll handle the rest.
        </p>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          "card border-2 border-dashed p-10 text-center cursor-pointer transition-all mb-6",
          isDragActive
            ? "border-slate-900 bg-slate-50"
            : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/50"
        )}
      >
        <input {...getInputProps()} />
        <div className="w-12 h-12 bg-slate-100 rounded-xl flex items-center justify-center mx-auto mb-4">
          <Upload className="w-5 h-5 text-slate-500" />
        </div>
        <p className="text-sm font-medium text-slate-800">
          {isDragActive ? "Drop files here" : "Drag & drop files, or click to browse"}
        </p>
        <p className="text-xs text-slate-400 mt-1.5">.txt · .csv · .json · .jsonl</p>
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="card overflow-hidden mb-6">
          <div className="px-4 py-2.5 border-b border-slate-100 flex items-center justify-between">
            <p className="text-xs font-medium text-slate-500">
              {files.length} file{files.length > 1 ? "s" : ""}
            </p>
          </div>
          <div className="divide-y divide-slate-50">
            {files.map((file) => {
              const status = statuses[file.name] || "ready";
              const Icon = getIcon(file.name);
              return (
                <div
                  key={file.name}
                  className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50/80 transition-colors"
                >
                  <div className="w-8 h-8 bg-slate-100 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Icon className="w-3.5 h-3.5 text-slate-500" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-slate-800 truncate font-mono">{file.name}</p>
                    <p className="text-xs text-slate-400">{fmtSize(file.size)}</p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {status === "done" && <CheckCircle className="w-4 h-4 text-emerald-500" />}
                    {status === "error" && <AlertCircle className="w-4 h-4 text-red-500" />}
                    {status === "uploading" && (
                      <span className="w-4 h-4 border-2 border-slate-300 border-t-slate-700 rounded-full animate-spin" />
                    )}
                    {status === "ready" && !uploading && (
                      <button
                        onClick={() => remove(file.name)}
                        className="text-slate-300 hover:text-red-400 transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3">
        {files.length > 0 && !allDone && (
          <button onClick={handleUpload} disabled={uploading} className="btn-primary">
            {uploading ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Uploading...
              </>
            ) : (
              <>
                <Upload className="w-3.5 h-3.5" />
                Upload {files.length} file{files.length > 1 ? "s" : ""}
              </>
            )}
          </button>
        )}
        {allDone && datasetId && (
          <button
            onClick={() => navigate(`/dashboard/dataset?id=${datasetId}`)}
            className="btn-primary"
          >
            Process Dataset <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
        {files.length === 0 && (
          <p className="text-sm text-slate-400">Add at least one file to continue</p>
        )}
      </div>
    </div>
  );
}
