import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { kataApi, label } from "@/lib/kata";
import { DifficultyPill, KindIcon, StatusIcon } from "./bits";
import { cn } from "@/lib/utils";

/** Every module and problem, with status. Used in the workspace's slide-out list. */
export function ProblemList({ currentId, onPick }: { currentId?: string; onPick?: () => void }) {
  const { data } = useQuery({ queryKey: ["kata-catalog"], queryFn: kataApi.catalog });
  if (!data) return <div className="p-6 text-sm text-muted-foreground">Loading…</div>;
  return (
    <div className="space-y-5">
      {data.lessons.map((l) => [
        data.lessons.length > 1 && (
          <div key={`lesson-${l.id}`} className="px-2 pt-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{l.title}</div>
        ),
        ...l.modules.map((m) => {
          const solved = m.problems.filter((p) => p.status === "solved").length;
          return (
            <div key={m.id}>
              <div className="mb-1.5 flex items-baseline justify-between px-2">
                <div className="text-[13px] font-semibold text-foreground">
                  <span className="mr-2 font-mono text-primary">{m.id}</span>
                  {m.title}
                </div>
                <span className="font-mono text-[11px] text-muted-foreground">{solved}/{m.problems.length}</span>
              </div>
              <div>
                {m.problems.map((p) => (
                  <Link
                    key={p.id}
                    to={`/kata/${p.id}`}
                    onClick={onPick}
                    className={cn(
                      "flex items-center gap-2.5 rounded-md px-2 py-1.5 text-[13px] transition-colors",
                      p.id === currentId ? "bg-primary/10 text-foreground" : "text-foreground/80 hover:bg-white/[0.04]",
                    )}
                  >
                    <StatusIcon status={p.status} className="h-3.5 w-3.5" />
                    <span className="w-9 font-mono text-[11.5px] text-muted-foreground">{label(p.id)}</span>
                    <span className="flex-1 truncate">{p.title}</span>
                    {(p.reviewing || (p.card && new Date(p.card.due).getTime() <= Date.now())) && (
                      <span className="rounded-full bg-sky-500/20 px-1.5 py-[1px] text-[10px] font-semibold text-sky-200">
                        {p.reviewing ? "reviewing" : "due"}
                      </span>
                    )}
                    <KindIcon kind={p.kind} />
                    <DifficultyPill d={p.difficulty} className="text-[10px]" />
                  </Link>
                ))}
              </div>
            </div>
          );
        }),
      ])}
    </div>
  );
}
