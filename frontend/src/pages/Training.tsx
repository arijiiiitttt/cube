import { useState, useEffect, useRef } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Square,
  ArrowRight,
  AlertCircle,
  Activity,
  Clock,
  TrendingDown,
  RefreshCw,
} from "lucide-react";
import toast from "react-hot-toast";
import { getTrainingStatus, getTrainingLogs, stopTraining, listModels } from "../lib/api";
import { cn, fmtTime } from "../lib/utils";

type DataPoint = { step: number; trainLoss: number; valLoss?: number };

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-2.5 text-xs shadow-card">
      <p className="text-slate-500 mb-1">Step {label}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(4)}
        </p>
      ))}
    </div>
  );
};

export default function TrainingPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const jobId = params.get("jobId");

  const [job, setJob] = useState<any>(null);
  const [chart, setChart] = useState<DataPoint[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [modelId, setModelId] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const prevStep = useRef(0);
  const elapsedRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  useEffect(() => {
    if (job?.status === "running") {
      elapsedRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    } else {
      clearInterval(elapsedRef.current);
    }
    return () => clearInterval(elapsedRef.current);
  }, [job?.status]);

  useEffect(() => {
    if (job?.status !== "done" || modelId || !jobId) return;
    const fetchModelId = async () => {
      try {
        const res = await listModels();
        const found = (res.data as any[]).find((m: any) => m.job_id === jobId);
        if (found) setModelId(found.id);
      } catch {
        /* ignore */
      }
    };
    fetchModelId();
    const t = setInterval(fetchModelId, 3000);
    return () => clearInterval(t);
  }, [job?.status, modelId, jobId]);

  useEffect(() => {
    if (!jobId) return;
    const poll = async () => {
      try {
        const [statusRes, logsRes] = await Promise.all([
          getTrainingStatus(jobId),
          getTrainingLogs(jobId),
        ]);
        const d = statusRes.data;
        setJob(d);
        setError(null);

        if (d.metrics?.length) {
          const points: DataPoint[] = d.metrics.map((m: any) => ({
            step: m.step,
            trainLoss: m.train_loss ?? m.loss,
            valLoss: m.eval_loss ?? m.val_loss,
          }));
          setChart(points);
          if (points.length && points[points.length - 1].step > prevStep.current) {
            prevStep.current = points[points.length - 1].step;
          }
        }

        if (logsRes.data?.logs) {
          setLogs(Array.isArray(logsRes.data.logs) ? logsRes.data.logs : logsRes.data.logs.split("\n"));
        }
      } catch (err: any) {
        setError(err.response?.data?.detail || "Failed to fetch status");
      }
    };
    poll();
    const interval = setInterval(poll, 2500);
    return () => clearInterval(interval);
  }, [jobId]);

  const handleStop = async () => {
    if (!jobId) return;
    try {
      await stopTraining(jobId);
      toast.success("Stop signal sent");
    } catch {
      toast.error("Failed to stop job");
    }
  };

  const isRunning = job?.status === "running" || job?.status === "pending";
  const isDone = job?.status === "done" || job?.status === "completed";
  const isError = job?.status === "error" || job?.status === "failed";

  if (!jobId) {
    return (
      <div className="animate-slide-up">
        <h1 className="page-title mb-2">Training</h1>
        <p className="page-desc mb-6">No active job. Start a training run from the Model page.</p>
        <button onClick={() => navigate("/dashboard/model")} className="btn-primary">
          Choose Model
        </button>
      </div>
    );
  }

  return (
    <div className="animate-slide-up">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="page-title">Training</h1>
          <p className="page-desc font-mono text-xs mt-1">Job {jobId.slice(0, 8)}…</p>
        </div>
        <div className="flex items-center gap-2">
          {isRunning && <span className="badge-green"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" /> Running</span>}
          {isDone && <span className="badge-blue">Done</span>}
          {isError && <span className="badge-red">Failed</span>}
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[
          { icon: Activity, label: "Status", value: job?.status || "…" },
          { icon: TrendingDown, label: "Train Loss", value: chart.length ? chart[chart.length - 1].trainLoss?.toFixed(4) : "—" },
          { icon: Clock, label: "Elapsed", value: fmtTime(elapsed) },
          { icon: Activity, label: "Step", value: chart.length ? chart[chart.length - 1].step : "—" },
        ].map((m) => (
          <div key={m.label} className="metric-card">
            <div className="flex items-center gap-1.5 mb-1">
              <m.icon className="w-3.5 h-3.5 text-slate-400" />
              <p className="text-xs text-slate-400">{m.label}</p>
            </div>
            <p className="text-lg font-semibold text-slate-900 font-mono capitalize">{m.value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-5 gap-4 mb-6">
        {/* Chart */}
        <div className="col-span-3 card p-4">
          <p className="text-xs font-medium text-slate-500 mb-3">Loss curve</p>
          <div className="h-[240px]">
            {chart.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="step" tick={{ fontSize: 11 }} stroke="#94a3b8" />
                  <YAxis tick={{ fontSize: 11 }} stroke="#94a3b8" domain={["auto", "auto"]} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="trainLoss" name="Train" stroke="#0f172a" strokeWidth={2} dot={false} />
                  {chart.some((d) => d.valLoss != null) && (
                    <Line type="monotone" dataKey="valLoss" name="Val" stroke="#3b82f6" strokeWidth={2} dot={false} />
                  )}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-slate-300">
                Waiting for metrics...
              </div>
            )}
          </div>
        </div>

        {/* Logs */}
        <div className="col-span-2 card overflow-hidden flex flex-col">
          <div className="px-4 py-2 border-b border-slate-100 flex items-center justify-between">
            <p className="text-xs font-mono text-slate-500">train.log</p>
            {isRunning && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />}
          </div>
          <div ref={logRef} className="flex-1 p-3 font-mono text-xs space-y-0.5 h-[240px] overflow-y-auto bg-slate-50">
            {logs.length === 0 ? (
              <p className="text-slate-300">No logs yet...</p>
            ) : (
              logs.map((l, i) => (
                <p
                  key={i}
                  className={cn(
                    l.includes("[DONE]")
                      ? "text-emerald-600"
                      : l.includes("[WARN]")
                      ? "text-amber-600"
                      : l.includes("[ERROR]")
                      ? "text-red-600"
                      : l.includes("[EPOCH]")
                      ? "text-slate-900 font-semibold"
                      : l.includes("[STEP]")
                      ? "text-slate-500"
                      : "text-slate-400"
                  )}
                >
                  {l}
                </p>
              ))
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-sm text-red-700">
          <AlertCircle className="w-4 h-4 flex-shrink-0" /> {error}
        </div>
      )}
      {isError && (
        <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-sm text-red-700">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          Training failed: {job?.error_msg || "Unknown error"}
        </div>
      )}

      <div className="flex gap-3">
        {isRunning && (
          <button onClick={handleStop} className="btn-danger">
            <Square className="w-3.5 h-3.5" /> Stop Training
          </button>
        )}
        {isDone && modelId && (
          <button
            onClick={() => navigate(`/dashboard/evaluation?modelId=${modelId}`)}
            className="btn-primary"
          >
            Evaluate Model <ArrowRight className="w-3.5 h-3.5" />
          </button>
        )}
        {isDone && !modelId && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Registering model...
          </div>
        )}
      </div>
    </div>
  );
}
