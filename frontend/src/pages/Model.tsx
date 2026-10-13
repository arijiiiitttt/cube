import { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { Brain, Zap, ArrowRight, Check, AlertCircle, Loader2, Sparkles } from "lucide-react";
import toast from "react-hot-toast";
import { startTraining, estimateTraining } from "../lib/api";
import { cn } from "../lib/utils";

const BASE_MODELS = [
  {
    id: "meta-llama/Llama-3.2-1B-Instruct",
    name: "Llama 3.2 1B",
    params: "1B",
    recommended: true,
    desc: "Best default for QLoRA on a single consumer/cloud GPU. Fast iteration.",
    tags: ["instruct", "rope", "gqa"],
  },
  {
    id: "Qwen/Qwen2.5-1.5B-Instruct",
    name: "Qwen2.5 1.5B",
    params: "1.5B",
    recommended: false,
    desc: "Strong on multilingual and code tasks for its size.",
    tags: ["instruct", "multilingual", "code"],
  },
  {
    id: "google/gemma-2-2b-it",
    name: "Gemma 2 2B",
    params: "2B",
    recommended: false,
    desc: "Good quality/size tradeoff; well-supported by Flash Attention 2.",
    tags: ["instruct", "gqa"],
  },
  {
    id: "meta-llama/Llama-3.1-8B-Instruct",
    name: "Llama 3.1 8B",
    params: "8B",
    recommended: false,
    desc: "Best quality ceiling — needs a 24GB+ GPU even with 4-bit QLoRA.",
    tags: ["instruct", "rope", "gqa"],
  },
];

export default function ModelPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = params.get("datasetId") || "";
  const datasetTokens = Number(params.get("tokens") || 50000);

  const [selected, setSelected] = useState(BASE_MODELS[0].id);
  const [config, setConfig] = useState({
    epochs: 3,
    per_device_batch_size: 4,
    gradient_accumulation_steps: 4,
    learning_rate: 0.0002,
    max_seq_len: 1024,
    lora_r: 16,
    lora_alpha: 32,
  });
  const [speed, setSpeed] = useState({
    load_in_4bit: true,
    use_lora: true,
    use_flash_attention_2: true,
    use_torch_compile: true,
    use_unsloth: false,
  });
  const [estimate, setEstimate] = useState<{ estimated_minutes: number; note: string } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        const res = await estimateTraining({
          base_model: selected,
          dataset_total_tokens: datasetTokens,
          epochs: config.epochs,
          per_device_batch_size: config.per_device_batch_size,
          gradient_accumulation_steps: config.gradient_accumulation_steps,
          max_seq_len: config.max_seq_len,
        });
        setEstimate(res.data);
      } catch {
        setEstimate(null);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [selected, config, datasetTokens]);

  const handleStart = async () => {
    if (!datasetId) {
      toast.error("No dataset ID");
      return;
    }
    setLoading(true);
    try {
      const res = await startTraining({
        dataset_id: datasetId,
        base_model: selected,
        ...config,
        ...speed,
      });
      const jobId = res.data.job_id || res.data.id;
      toast.success("Training started!");
      navigate(`/dashboard/training?jobId=${jobId}`);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Failed to start training");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <h1 className="page-title">Choose Model & Config</h1>
        <p className="page-desc">Select a base model and tune QLoRA hyperparameters.</p>
      </div>

      {/* Model cards */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        {BASE_MODELS.map((m) => (
          <button
            key={m.id}
            onClick={() => setSelected(m.id)}
            className={cn(
              "card p-4 text-left transition-all relative",
              selected === m.id
                ? "ring-2 ring-slate-900 border-slate-900"
                : "hover:border-slate-300"
            )}
          >
            {m.recommended && (
              <span className="absolute top-3 right-3 badge-blue text-[10px]">
                <Sparkles className="w-2.5 h-2.5" /> Recommended
              </span>
            )}
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-4 h-4 text-slate-500" />
              <span className="text-sm font-semibold text-slate-900">{m.name}</span>
              <span className="badge-gray text-[10px]">{m.params}</span>
            </div>
            <p className="text-xs text-slate-500 leading-relaxed mb-2">{m.desc}</p>
            <div className="flex flex-wrap gap-1">
              {m.tags.map((t) => (
                <span key={t} className="badge-gray text-[10px]">
                  {t}
                </span>
              ))}
            </div>
            {selected === m.id && (
              <div className="absolute top-3 left-3">
                <Check className="w-4 h-4 text-slate-900" />
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Hyperparams */}
      <div className="card p-5 mb-5">
        <h2 className="text-sm font-semibold text-slate-900 mb-4">Hyperparameters</h2>
        <div className="grid grid-cols-2 gap-x-8 gap-y-4">
          {[
            { key: "epochs", label: "Epochs", min: 1, max: 10, step: 1, fmt: (v: number) => v.toString() },
            { key: "per_device_batch_size", label: "Batch Size / GPU", min: 1, max: 32, step: 1, fmt: (v: number) => v.toString() },
            { key: "gradient_accumulation_steps", label: "Grad Accum Steps", min: 1, max: 32, step: 1, fmt: (v: number) => v.toString() },
            { key: "learning_rate", label: "Learning Rate", min: 0.00002, max: 0.001, step: 0.00002, fmt: (v: number) => v.toExponential(1) },
            { key: "max_seq_len", label: "Seq Length", min: 256, max: 4096, step: 256, fmt: (v: number) => v.toString() },
            { key: "lora_r", label: "LoRA Rank (r)", min: 4, max: 128, step: 4, fmt: (v: number) => v.toString() },
          ].map(({ key, label, min, max, step, fmt }) => (
            <div key={key}>
              <div className="flex justify-between mb-1.5">
                <label className="text-xs text-slate-600 font-medium">{label}</label>
                <span className="text-xs font-mono text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded">
                  {fmt((config as any)[key])}
                </span>
              </div>
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={(config as any)[key]}
                onChange={(e) => setConfig((c) => ({ ...c, [key]: +e.target.value }))}
                className="w-full accent-slate-900 h-1"
              />
            </div>
          ))}
        </div>

        {/* Speed toggles */}
        <div className="mt-5 pt-4 border-t border-slate-100">
          <p className="text-xs font-medium text-slate-600 mb-3">Optimizations</p>
          <div className="flex flex-wrap gap-2">
            {(
              [
                ["load_in_4bit", "4-bit Quantization"],
                ["use_lora", "LoRA"],
                ["use_flash_attention_2", "Flash Attention 2"],
                ["use_torch_compile", "torch.compile"],
                ["use_unsloth", "Unsloth"],
              ] as const
            ).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setSpeed((s) => ({ ...s, [key]: !s[key] }))}
                className={cn(
                  "text-xs px-3 py-1.5 rounded-full border transition-all font-medium",
                  speed[key]
                    ? "bg-slate-900 text-white border-slate-900"
                    : "bg-white text-slate-500 border-slate-200 hover:border-slate-300"
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-5 pt-4 border-t border-slate-100 flex items-center gap-2 text-xs text-slate-400">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          {estimate
            ? `Estimated: ~${estimate.estimated_minutes} min on an A10G — heuristic`
            : "Estimating..."}
        </div>
      </div>

      {!datasetId && (
        <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-lg p-3 mb-4 text-xs text-amber-700">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          No dataset ID — process a dataset first
        </div>
      )}

      <button onClick={handleStart} disabled={loading || !datasetId} className="btn-primary">
        {loading ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Starting...
          </>
        ) : (
          <>
            Start Training <ArrowRight className="w-3.5 h-3.5" />
          </>
        )}
      </button>
    </div>
  );
}
