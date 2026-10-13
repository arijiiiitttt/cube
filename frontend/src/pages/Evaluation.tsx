import { useState, useEffect, useRef } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import {
  FlaskConical,
  Send,
  Loader2,
  AlertCircle,
  ArrowRight,
  CheckCircle,
  TrendingDown,
  BarChart2,
  RefreshCw,
} from "lucide-react";
import toast from "react-hot-toast";
import { runEvaluation, getEvalResults, generateText } from "../lib/api";
import { cn } from "../lib/utils";

export default function EvaluationPage() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const modelId = params.get("modelId") || "";

  const [phase, setPhase] = useState<"idle" | "running" | "done" | "error">("idle");
  const [results, setResults] = useState<any>(null);
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [genLoading, setGen] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  const startPolling = () => {
    let elapsed = 0;
    pollRef.current = setInterval(async () => {
      elapsed += 2;
      setProgress(Math.min(95, elapsed * 3));
      try {
        const res = await getEvalResults(modelId);
        const d = res.data;
        if (d.perplexity && d.perplexity > 0 && !d.error) {
          clearInterval(pollRef.current);
          setResults(d);
          setPhase("done");
          setProgress(100);
          toast.success("Evaluation complete");
        } else if (d.error) {
          clearInterval(pollRef.current);
          setPhase("error");
          setErrorMsg(d.error);
        }
      } catch {
        /* keep polling */
      }
      if (elapsed > 120) {
        clearInterval(pollRef.current);
        setPhase("error");
        setErrorMsg("Evaluation timed out");
      }
    }, 2000);
  };

  useEffect(() => () => clearInterval(pollRef.current), []);

  const runEval = async () => {
    if (!modelId) {
      toast.error("No model ID — complete training first");
      return;
    }
    setPhase("running");
    setProgress(0);
    setErrorMsg("");
    try {
      await runEvaluation(modelId);
      startPolling();
    } catch (err: any) {
      setPhase("error");
      setErrorMsg(err.response?.data?.detail || "Failed to start evaluation");
    }
  };

  const generate = async () => {
    if (!prompt.trim() || !modelId) return;
    setGen(true);
    setOutput("");
    try {
      const res = await generateText(modelId, {
        prompt,
        max_new_tokens: 256,
        temperature: 0.7,
      });
      setOutput(res.data.text || res.data.generated_text || JSON.stringify(res.data));
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Generation failed");
    } finally {
      setGen(false);
    }
  };

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <h1 className="page-title">Evaluation</h1>
        <p className="page-desc">Score perplexity and try interactive generation.</p>
      </div>

      {!modelId && (
        <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-xl p-4 mb-6 text-sm text-amber-700">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          No model ID. Finish a training run first.
        </div>
      )}

      {/* Eval controls */}
      <div className="card p-5 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <FlaskConical className="w-4 h-4 text-slate-500" />
            <h2 className="text-sm font-semibold text-slate-900">Perplexity Evaluation</h2>
          </div>
          {phase === "idle" && (
            <button onClick={runEval} disabled={!modelId} className="btn-primary">
              <BarChart2 className="w-3.5 h-3.5" /> Run Evaluation
            </button>
          )}
          {phase === "running" && (
            <button disabled className="btn-primary">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Evaluating...
            </button>
          )}
          {phase === "done" && (
            <span className="badge-green">
              <CheckCircle className="w-3 h-3" /> Complete
            </span>
          )}
        </div>

        {phase === "running" && (
          <div className="mb-4">
            <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-slate-900 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-xs text-slate-400 mt-1.5">Scoring validation set...</p>
          </div>
        )}

        {results && (
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-slate-50 rounded-lg p-3">
              <div className="flex items-center gap-1.5 mb-1">
                <TrendingDown className="w-3.5 h-3.5 text-slate-400" />
                <p className="text-xs text-slate-400">Perplexity</p>
              </div>
              <p className="text-xl font-semibold font-mono text-slate-900">
                {results.perplexity?.toFixed(2)}
              </p>
            </div>
            <div className="bg-slate-50 rounded-lg p-3">
              <p className="text-xs text-slate-400 mb-1">Samples scored</p>
              <p className="text-xl font-semibold font-mono text-slate-900">
                {results.n_samples || results.samples || "—"}
              </p>
            </div>
            <div className="bg-slate-50 rounded-lg p-3">
              <p className="text-xs text-slate-400 mb-1">Loss</p>
              <p className="text-xl font-semibold font-mono text-slate-900">
                {results.loss?.toFixed(4) || "—"}
              </p>
            </div>
          </div>
        )}

        {phase === "error" && (
          <div className="flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Evaluation failed</p>
              <p className="text-xs mt-0.5">{errorMsg}</p>
              <button onClick={runEval} className="mt-2 text-xs font-medium underline">
                <RefreshCw className="w-3 h-3 inline mr-1" /> Retry
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Playground */}
      <div className="card p-5 mb-6">
        <h2 className="text-sm font-semibold text-slate-900 mb-4">Quick Generation</h2>
        <div className="space-y-3">
          <textarea
            className="input min-h-[80px] resize-y"
            placeholder="Enter a prompt to test the model..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <button
            onClick={generate}
            disabled={genLoading || !prompt.trim() || !modelId}
            className="btn-primary"
          >
            {genLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> Generating...
              </>
            ) : (
              <>
                <Send className="w-3.5 h-3.5" /> Generate
              </>
            )}
          </button>
          {output && (
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mt-2">
              <p className="text-xs text-slate-400 mb-1.5 font-mono">output</p>
              <p className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">{output}</p>
            </div>
          )}
        </div>
      </div>

      {phase === "done" && modelId && (
        <button
          onClick={() => navigate(`/dashboard/deploy?modelId=${modelId}`)}
          className="btn-primary"
        >
          Deploy Model <ArrowRight className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}
