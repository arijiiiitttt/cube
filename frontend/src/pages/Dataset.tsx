import { useState, useEffect, useRef } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import {
  CheckCircle,
  Circle,
  Loader2,
  ArrowRight,
  AlertCircle,
  Database,
  RefreshCw,
} from "lucide-react";
import toast from "react-hot-toast";
import { processDataset, getDatasetStatus } from "../lib/api";
import { cn } from "../lib/utils";

const STEPS = [
  { label: "Cleaning", desc: "Remove HTML, URLs, noise" },
  { label: "Deduplication", desc: "Remove near-duplicate lines" },
  { label: "Tokenization", desc: "BPE tokenizer, vocab 32K" },
  { label: "Train/Val Split", desc: "90% train · 10% validation" },
];

export default function DatasetPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = params.get("id");

  const [status, setStatus] = useState<"idle" | "processing" | "ready" | "error">("idle");
  const [stats, setStats] = useState<any>(null);
  const [log, setLog] = useState<string[]>([]);
  const [errorMsg, setError] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval>>();
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => clearInterval(pollRef.current), []);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const addLog = (msg: string) => setLog((p) => [...p.slice(-60), msg]);

  const startPolling = () => {
    addLog("[INFO] Processing started on server...");
    let step = 0;
    pollRef.current = setInterval(async () => {
      step = Math.min(3, step + 1);
      setActiveStep(step);
      try {
        const res = await getDatasetStatus(datasetId!);
        const d = res.data;
        if (d.status === "ready") {
          clearInterval(pollRef.current);
          setStats(d);
          setStatus("ready");
          setActiveStep(4);
          addLog(`[DONE] ${d.total_tokens?.toLocaleString() || "?"} tokens ready`);
          toast.success("Dataset ready!");
        } else if (d.status === "error") {
          clearInterval(pollRef.current);
          setStatus("error");
          setError("Processing failed on server");
          addLog("[ERROR] Processing failed");
        } else {
          addLog(`[INFO] Status: ${d.status}...`);
        }
      } catch {
        addLog("[WARN] Poll failed — retrying...");
      }
    }, 2000);
  };

  const start = async () => {
    if (!datasetId) {
      toast.error("No dataset ID — upload files first");
      return;
    }
    setStatus("processing");
    setError("");
    setLog([]);
    setActiveStep(0);
    addLog("[INFO] Submitting process request...");
    try {
      await processDataset(datasetId);
      startPolling();
    } catch (err: any) {
      setStatus("error");
      setError(err.response?.data?.detail || "Failed to start processing");
    }
  };

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <h1 className="page-title">Process Dataset</h1>
        <p className="page-desc">
          Clean, dedupe, tokenize and split your data for training.
        </p>
      </div>

      {!datasetId && (
        <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-xl p-4 mb-6 text-sm text-amber-700">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          No dataset ID found. Go back and upload files first.
        </div>
      )}

      {/* Steps */}
      <div className="card p-5 mb-5">
        <div className="space-y-3">
          {STEPS.map((s, i) => {
            const done = status === "ready" || i < activeStep;
            const active = status === "processing" && i === activeStep;
            return (
              <div key={s.label} className="flex items-center gap-3">
                <div className="flex-shrink-0">
                  {done ? (
                    <CheckCircle className="w-5 h-5 text-emerald-500" />
                  ) : active ? (
                    <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                  ) : (
                    <Circle className="w-5 h-5 text-slate-300" />
                  )}
                </div>
                <div className="flex-1">
                  <p className={cn("text-sm font-medium", done || active ? "text-slate-900" : "text-slate-400")}>
                    {s.label}
                  </p>
                  <p className="text-xs text-slate-400">{s.desc}</p>
                </div>
                {active && <span className="text-xs text-blue-500 font-medium">Running...</span>}
              </div>
            );
          })}
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-3 gap-3 mb-5">
          {[
            { label: "Total tokens", value: stats.total_tokens?.toLocaleString() || "—" },
            { label: "Train samples", value: stats.train_samples?.toLocaleString() || "—" },
            { label: "Val samples", value: stats.val_samples?.toLocaleString() || "—" },
          ].map((m) => (
            <div key={m.label} className="metric-card">
              <p className="text-xs text-slate-400 mb-1">{m.label}</p>
              <p className="text-lg font-semibold text-slate-900 font-mono">{m.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Log */}
      {log.length > 0 && (
        <div className="card overflow-hidden mb-5">
          <div className="px-4 py-2 border-b border-slate-100 flex items-center justify-between">
            <p className="text-xs font-mono text-slate-500">server log</p>
            {status === "processing" && (
              <RefreshCw className="w-3 h-3 text-slate-400 animate-spin" />
            )}
          </div>
          <div
            ref={logRef}
            className="p-3 font-mono text-xs space-y-0.5 max-h-40 overflow-y-auto bg-slate-50"
          >
            {log.map((line, i) => (
              <p
                key={i}
                className={cn(
                  line.includes("[DONE]")
                    ? "text-emerald-600"
                    : line.includes("[ERROR]")
                    ? "text-red-600"
                    : line.includes("[WARN]")
                    ? "text-amber-600"
                    : "text-slate-500"
                )}
              >
                {line}
              </p>
            ))}
          </div>
        </div>
      )}

      {status === "error" && (
        <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-xl p-4 mb-5">
          <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-red-700">Processing failed</p>
            <p className="text-xs text-red-600 mt-0.5">{errorMsg}</p>
          </div>
        </div>
      )}

      <div className="flex gap-3">
        {status === "idle" && (
          <button onClick={start} disabled={!datasetId} className="btn-primary">
            <Database className="w-3.5 h-3.5" /> Process Dataset
          </button>
        )}
        {status === "processing" && (
          <button disabled className="btn-primary">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Processing...
          </button>
        )}
        {status === "ready" && (
          <button
            onClick={() =>
              navigate(
                `/dashboard/model?datasetId=${datasetId}&tokens=${stats?.total_tokens || 50000}`
              )
            }
            className="btn-primary"
          >
            Choose Model <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
        {status === "error" && (
          <button onClick={start} className="btn-secondary">
            <RefreshCw className="w-3.5 h-3.5" /> Retry
          </button>
        )}
      </div>
    </div>
  );
}
