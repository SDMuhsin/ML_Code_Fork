import { CheckCircle2, Circle, CircleDot, Cpu, Eye, Zap } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Device, Difficulty, KataEnv, KataStatus } from "@/lib/kata";
import { fmtBytes } from "@/lib/kata";
import { cn } from "@/lib/utils";

export function StatusIcon({ status, className }: { status: KataStatus; className?: string }) {
  if (status === "solved") return <CheckCircle2 className={cn("h-4 w-4 text-emerald-400", className)} />;
  if (status === "attempted") return <CircleDot className={cn("h-4 w-4 text-amber-400", className)} />;
  return <Circle className={cn("h-4 w-4 text-white/15", className)} />;
}

const diffStyle: Record<Difficulty, string> = {
  Easy: "text-emerald-400 bg-emerald-400/10 border-emerald-400/20",
  Medium: "text-amber-400 bg-amber-400/10 border-amber-400/20",
  Hard: "text-rose-400 bg-rose-400/10 border-rose-400/20",
};

export function DifficultyPill({ d, className }: { d: Difficulty; className?: string }) {
  return (
    <span className={cn("inline-flex items-center rounded-full border px-2 py-[1px] text-[11px] font-medium", diffStyle[d], className)}>
      {d}
    </span>
  );
}

export function KindIcon({ kind, className }: { kind: "code" | "predict"; className?: string }) {
  if (kind !== "predict") return null;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Eye className={cn("h-3.5 w-3.5 text-sky-400/80", className)} />
      </TooltipTrigger>
      <TooltipContent>Predict: write the shapes before you run anything</TooltipContent>
    </Tooltip>
  );
}

/** Green when a CUDA device is ready, amber while workers warm up, red if Python is broken. */
export function GpuChip({ env, compact }: { env?: KataEnv; compact?: boolean }) {
  const e = env?.env;
  let dot = "bg-amber-400 animate-pulse";
  let text = "Starting Python…";
  if (env?.lastError && !e) {
    dot = "bg-rose-500";
    text = "Python worker failed";
  } else if (e) {
    dot = e.cuda_available ? "bg-emerald-400" : "bg-slate-400";
    text = e.cuda_available ? (e.device_name || "CUDA").replace("NVIDIA GeForce ", "") : "CPU only";
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex cursor-default items-center gap-2 rounded-md border border-white/[0.06] bg-white/[0.03] px-2.5 py-1 text-xs text-muted-foreground">
          <span className={cn("h-2 w-2 rounded-full", dot)} />
          {!compact && <span className="max-w-[180px] truncate">{text}</span>}
        </div>
      </TooltipTrigger>
      <TooltipContent className="max-w-sm text-xs leading-relaxed">
        {e ? (
          <div className="space-y-0.5 font-mono">
            <div>torch {e.torch} · python {e.python}</div>
            {e.cuda_available ? (
              <div>
                {e.device_name} · sm_{e.capability?.replace(".", "")} · {fmtBytes(e.total_memory)} · CUDA {e.cuda_version}
              </div>
            ) : (
              <div>torch.cuda.is_available() is False</div>
            )}
            <div className="text-muted-foreground">{env?.ready ?? 0} warm worker(s) ready</div>
          </div>
        ) : (
          <div className="whitespace-pre-wrap font-mono">{env?.lastError || "Importing torch in the background…"}</div>
        )}
      </TooltipContent>
    </Tooltip>
  );
}

export function DeviceToggle({
  device,
  onChange,
  env,
  allowed,
}: {
  device: Device;
  onChange: (d: Device) => void;
  env?: KataEnv;
  allowed: Device[];
}) {
  const cudaOk = Boolean(env?.env?.cuda_available) && allowed.includes("cuda");
  const why = !env?.env?.cuda_available
    ? "torch cannot see a CUDA device"
    : !allowed.includes("cuda")
      ? "this problem's checks are CPU-specific"
      : `run on ${env?.env?.device_name}`;
  const opt = (d: Device, enabled: boolean, icon: JSX.Element, tip: string) => (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          disabled={!enabled}
          onClick={() => onChange(d)}
          className={cn(
            "flex items-center gap-1.5 rounded-[5px] px-2.5 py-1 text-xs font-medium transition-colors",
            device === d ? "bg-white/10 text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
            !enabled && "cursor-not-allowed opacity-40 hover:text-muted-foreground",
          )}
        >
          {icon}
          {d.toUpperCase()}
        </button>
      </TooltipTrigger>
      <TooltipContent>{tip}</TooltipContent>
    </Tooltip>
  );
  return (
    <div className="flex items-center rounded-md border border-white/[0.06] bg-white/[0.03] p-0.5">
      {opt("cpu", true, <Cpu className="h-3.5 w-3.5" />, "Run on the CPU (the curriculum's default)")}
      {opt("cuda", cudaOk, <Zap className="h-3.5 w-3.5" />, why)}
    </div>
  );
}
