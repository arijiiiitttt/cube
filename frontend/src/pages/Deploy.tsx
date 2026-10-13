import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Rocket,
  Key,
  Copy,
  Check,
  AlertCircle,
  Loader2,
  ExternalLink,
  Send,
} from "lucide-react";
import toast from "react-hot-toast";
import { deployModel, getDeployInfo, generateText } from "../lib/api";
import { cn } from "../lib/utils";

export default function DeployPage() {
  const [params] = useSearchParams();
  const modelId = params.get("modelId") || "";

  const [deploying, setDeploying] = useState(false);
  const [info, setInfo] = useState<any>(null);
  const [copied, setCopied] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [output, setOutput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  useEffect(() => {
    if (!modelId) return;
    getDeployInfo(modelId)
      .then((res) => {
        if (res.data?.api_key || res.data?.endpoint) setInfo(res.data);
      })
      .catch(() => {});
  }, [modelId]);

  const handleDeploy = async () => {
    if (!modelId) {
      toast.error("No model ID");
      return;
    }
    setDeploying(true);
    try {
      const res = await deployModel(modelId);
      setInfo(res.data);
      toast.success("Model deployed!");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Deploy failed");
    } finally {
      setDeploying(false);
    }
  };

  const copyKey = () => {
    if (!info?.api_key) return;
    navigator.clipboard.writeText(info.api_key);
    setCopied(true);
    toast.success("API key copied");
    setTimeout(() => setCopied(false), 2000);
  };

  const chat = async () => {
    if (!prompt.trim() || !modelId) return;
    setChatLoading(true);
    setOutput("");
    try {
      const res = await generateText(modelId, {
        prompt,
        max_new_tokens: 512,
        temperature: 0.7,
      });
      setOutput(res.data.text || res.data.generated_text || JSON.stringify(res.data));
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Generation failed");
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <h1 className="page-title">Deploy</h1>
        <p className="page-desc">Get an API key and chat with your fine-tuned model.</p>
      </div>

      {!modelId && (
        <div className="flex items-center gap-2 bg-amber-50 border border-amber-200 rounded-xl p-4 mb-6 text-sm text-amber-700">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          No model ID. Complete evaluation first.
        </div>
      )}

      {/* Deploy card */}
      <div className="card p-5 mb-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Rocket className="w-4 h-4 text-slate-500" />
            <h2 className="text-sm font-semibold text-slate-900">Deployment</h2>
          </div>
          {!info && (
            <button onClick={handleDeploy} disabled={deploying || !modelId} className="btn-primary">
              {deploying ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Deploying...
                </>
              ) : (
                <>
                  <Rocket className="w-3.5 h-3.5" /> Deploy Model
                </>
              )}
            </button>
          )}
          {info && <span className="badge-green">Live</span>}
        </div>

        {info && (
          <div className="space-y-4">
            <div>
              <label className="label flex items-center gap-1.5">
                <Key className="w-3.5 h-3.5" /> API Key
              </label>
              <div className="flex gap-2">
                <input
                  className="input font-mono text-xs"
                  value={info.api_key || "••••••••••••"}
                  readOnly
                />
                <button onClick={copyKey} className="btn-secondary flex-shrink-0">
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
            {info.endpoint && (
              <div>
                <label className="label">Endpoint</label>
                <div className="flex items-center gap-2">
                  <code className="text-xs font-mono bg-slate-50 border border-slate-200 rounded-lg px-3 py-2.5 flex-1 truncate">
                    {info.endpoint}
                  </code>
                  <a
                    href={info.endpoint}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-secondary flex-shrink-0"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>
            )}
            <div className="bg-slate-50 rounded-lg p-3 text-xs text-slate-500 font-mono leading-relaxed">
              <p className="text-slate-400 mb-1"># Example request</p>
              {`curl -X POST ${info.endpoint || "http://localhost:8000/inference/" + modelId + "/generate"} \\
  -H "Authorization: Bearer ${info.api_key || "YOUR_KEY"}" \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "Hello!", "max_new_tokens": 128}'`}
            </div>
          </div>
        )}
      </div>

      {/* Chat playground */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold text-slate-900 mb-4">Chat Playground</h2>
        <div className="space-y-3">
          <textarea
            className="input min-h-[100px] resize-y"
            placeholder="Type a message..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                chat();
              }
            }}
          />
          <button
            onClick={chat}
            disabled={chatLoading || !prompt.trim() || !modelId}
            className="btn-primary"
          >
            {chatLoading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" /> Thinking...
              </>
            ) : (
              <>
                <Send className="w-3.5 h-3.5" /> Send
              </>
            )}
          </button>
          {output && (
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
              <p className="text-xs text-slate-400 mb-1.5 font-mono">assistant</p>
              <p className="text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">{output}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
