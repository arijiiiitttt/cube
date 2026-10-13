import { Outlet, NavLink } from "react-router-dom";
import {
  Box,
  Upload,
  Database,
  Brain,
  Activity,
  FlaskConical,
  Rocket,
  ExternalLink,
  ChevronRight,
} from "lucide-react";
import { cn } from "../lib/utils";

const NAV = [
  { to: "/dashboard/upload", label: "Upload", icon: Upload, step: "1" },
  { to: "/dashboard/dataset", label: "Dataset", icon: Database, step: "2" },
  { to: "/dashboard/model", label: "Model", icon: Brain, step: "3" },
  { to: "/dashboard/training", label: "Training", icon: Activity, step: "4" },
  { to: "/dashboard/evaluation", label: "Evaluation", icon: FlaskConical, step: "5" },
  { to: "/dashboard/deploy", label: "Deploy", icon: Rocket, step: "6" },
];

const EXTERNAL = [
  { href: "http://localhost:5000", label: "MLflow" },
  { href: "http://localhost:5555", label: "Flower" },
  { href: "http://localhost:9001", label: "MinIO" },
  { href: "http://localhost:8000/docs", label: "API Docs" },
];

export default function DashboardLayout() {
  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      {/* Sidebar */}
      <aside className="w-[240px] flex-shrink-0 bg-white border-r border-slate-200 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center">
              <Box className="w-4 h-4 text-white" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-900 leading-none">Cube</p>
              <p className="text-[11px] text-slate-400 mt-0.5">LLM Fine-Tuning</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
          <p className="px-2 mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Pipeline
          </p>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm transition-all",
                  isActive
                    ? "bg-slate-900 text-white shadow-sm"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                )
              }
            >
              {({ isActive }) => (
                <>
                  <item.icon className={cn("w-4 h-4 flex-shrink-0", isActive ? "text-white" : "text-slate-400 group-hover:text-slate-600")} />
                  <span className="flex-1 font-medium">{item.label}</span>
                  <span className={cn("text-[10px] font-mono", isActive ? "text-slate-400" : "text-slate-300")}>
                    0{item.step}
                  </span>
                  {isActive && <ChevronRight className="w-3 h-3 opacity-60" />}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* External */}
        <div className="px-3 py-3 border-t border-slate-100">
          <p className="px-2 mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Tools
          </p>
          <div className="space-y-0.5">
            {EXTERNAL.map((link) => (
              <a
                key={link.href}
                href={link.href}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-2.5 py-1.5 text-xs text-slate-400 hover:text-slate-600 rounded-md hover:bg-slate-50 transition-colors"
              >
                <ExternalLink className="w-3 h-3" />
                {link.label}
              </a>
            ))}
          </div>
        </div>

      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-8 py-8 animate-fade-in">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
