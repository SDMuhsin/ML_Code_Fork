import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Check, Lock, RotateCcw } from "lucide-react";
import type { TreeNode } from "@/lib/kata";
import { cn } from "@/lib/utils";

// The learning path as a tree, NeetCode-roadmap style.
//
// Nothing here is specific to Lesson 1: the server sends every module with its
// `after:` edges and depth (content/progression.md). Each lesson is a band;
// inside a band, one row per depth. Rows wrap on narrow screens, and the edges
// are drawn after layout from where the nodes actually landed, so any number of
// lessons and modules lays itself out.

type Edge = { from: string; to: string; d: string; done: boolean };

/** Lesson bands, each a list of rows, ordered so that children sit under their parents. */
function layout(nodes: TreeNode[]) {
  const lessons: { id: string; title: string; rows: TreeNode[][] }[] = [];
  for (const n of nodes) {
    let l = lessons.find((x) => x.id === n.lesson);
    if (!l) lessons.push((l = { id: n.lesson, title: n.lessonTitle, rows: [] }));
  }
  const pos = new Map<string, number>(); // 0..1 horizontal position, for ordering the next row
  for (const l of lessons) {
    const mine = nodes.filter((n) => n.lesson === l.id);
    const depths = [...new Set(mine.map((n) => n.depth))].sort((a, b) => a - b);
    for (const d of depths) {
      const row = mine.filter((n) => n.depth === d);
      const bary = (n: TreeNode) => {
        const ps = n.after.map((a) => pos.get(a)).filter((x): x is number => x !== undefined);
        return ps.length ? ps.reduce((s, x) => s + x, 0) / ps.length : 0.5;
      };
      row.sort((a, b) => bary(a) - bary(b) || a.index - b.index);
      row.forEach((n, i) => pos.set(n.id, row.length === 1 ? 0.5 : i / (row.length - 1)));
      l.rows.push(row);
    }
  }
  return lessons;
}

const statusStyle: Record<TreeNode["status"], string> = {
  done: "border-emerald-500/40 bg-emerald-500/[0.08] hover:border-emerald-400/70",
  active: "border-primary/45 bg-primary/[0.07] hover:border-primary/80",
  ready: "border-white/[0.16] bg-[#161a24] hover:border-white/35",
  locked: "border-white/[0.06] bg-[#12151c] [&>*]:opacity-50 hover:[&>*]:opacity-90",
};

function NodeBox({ n, isNext, selected, onPick, onHover, refFn, i }: {
  n: TreeNode; isNext: boolean; selected: boolean; onPick: (id: string) => void; onHover: (id: string | null) => void;
  refFn: (el: HTMLButtonElement | null) => void; i: number;
}) {
  const pct = (n.solved / Math.max(1, n.total)) * 100;
  return (
    <motion.button
      ref={refFn}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: Math.min(i * 0.025, 0.35) }}
      onClick={() => onPick(n.id)}
      onMouseEnter={() => onHover(n.id)}
      onMouseLeave={() => onHover(null)}
      onFocus={() => onHover(n.id)}
      onBlur={() => onHover(null)}
      title={n.status === "locked" ? `Builds on ${n.after.join(", ")}. You can still open it.` : undefined}
      className={cn(
        "relative z-10 flex w-[150px] flex-col rounded-xl border px-3 pb-2.5 pt-2 text-left transition-colors sm:w-[172px]",
        statusStyle[n.status],
        isNext && "ring-2 ring-primary/70 ring-offset-2 ring-offset-[#0e1015]",
        selected && "ring-2 ring-white/50 ring-offset-2 ring-offset-[#0e1015]",
      )}
    >
      {isNext && (
        <span className="absolute -top-2.5 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-primary px-2 py-[1px] text-[10px] font-bold uppercase tracking-wide text-primary-foreground">
          Up next
        </span>
      )}
      <div className="flex items-center gap-1.5">
        <span className={cn(
          "font-mono text-[11px] font-bold",
          n.status === "done" ? "text-emerald-300" : n.status === "active" ? "text-primary" : "text-muted-foreground",
        )}>
          {n.id}
        </span>
        {n.status === "done" && <Check className="h-3 w-3 text-emerald-300" />}
        {n.status === "locked" && <Lock className="h-3 w-3 text-muted-foreground/70" />}
        {n.due > 0 && (
          <span className="ml-auto flex items-center gap-0.5 rounded-full bg-sky-500/20 px-1.5 py-[1px] text-[10px] font-semibold text-sky-200" title={`${n.due} due for review`}>
            <RotateCcw className="h-2.5 w-2.5" /> {n.due}
          </span>
        )}
      </div>
      <div className="mt-0.5 line-clamp-2 min-h-[2.5em] text-[12.5px] font-medium leading-tight text-foreground/90">{n.title}</div>
      <div className="mt-2 flex items-center gap-2">
        <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/[0.07]">
          <div className={cn("h-full rounded-full", n.status === "done" ? "bg-emerald-400" : "bg-primary/80")} style={{ width: `${pct}%` }} />
        </div>
        <span className="font-mono text-[10px] text-muted-foreground">{n.solved}/{n.total}</span>
      </div>
    </motion.button>
  );
}

export function Roadmap({ nodes, nextModule, selected, onPick }: {
  nodes: TreeNode[]; nextModule?: string | null; selected?: string | null; onPick: (id: string) => void;
}) {
  const bands = useMemo(() => layout(nodes), [nodes]);
  const wrap = useRef<HTMLDivElement>(null);
  const boxes = useRef(new Map<string, HTMLButtonElement>());
  const [edges, setEdges] = useState<Edge[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<string | null>(null);
  // The module you point at (else the one picked) shows its edges on top of the cards:
  // what it needs and what it unlocks. Other edges stay quiet behind the cards.
  const focus = hover ?? selected ?? null;
  const isHot = (e: Edge) => e.from === focus || e.to === focus || (!focus && e.to === nextModule);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const byId = new Map(nodes.map((n) => [n.id, n]));
    const measure = () => {
      const base = el.getBoundingClientRect();
      const out: Edge[] = [];
      for (const n of nodes) {
        const to = boxes.current.get(n.id);
        if (!to) continue;
        const b = to.getBoundingClientRect();
        for (const a of n.after) {
          const from = boxes.current.get(a);
          if (!from) continue;
          const r = from.getBoundingClientRect();
          const x1 = r.left + r.width / 2 - base.left;
          const y1 = r.bottom - base.top;
          const x2 = b.left + b.width / 2 - base.left;
          const y2 = b.top - base.top;
          const bend = Math.max(24, (y2 - y1) / 2);
          out.push({
            from: a, to: n.id,
            d: `M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`,
            done: byId.get(a)?.status === "done",
          });
        }
      }
      setEdges(out);
      setSize({ w: base.width, h: base.height });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    // Nodes animate in; measure once more when they settle.
    const t = setTimeout(measure, 450);
    return () => { ro.disconnect(); clearTimeout(t); };
  }, [nodes, bands]);

  return (
    <div ref={wrap} className="relative">
      <svg className="pointer-events-none absolute inset-0" width={size.w} height={size.h} aria-hidden>
        {edges.filter((e) => !isHot(e)).map((e) => (
          <path
            key={`${e.from}-${e.to}`}
            d={e.d}
            fill="none"
            strokeWidth={1.5}
            strokeDasharray={e.done ? undefined : "4 4"}
            className={cn(e.done ? "stroke-emerald-400/40" : "stroke-white/[0.14]", focus && "opacity-50")}
          />
        ))}
      </svg>
      <svg className="pointer-events-none absolute inset-0 z-20" width={size.w} height={size.h} aria-hidden>
        {edges.filter(isHot).map((e) => (
          <path
            key={`${e.from}-${e.to}`}
            d={e.d}
            fill="none"
            strokeWidth={2}
            strokeDasharray={e.done ? undefined : "5 4"}
            className={e.done ? "stroke-emerald-300/90" : "stroke-primary/90"}
          />
        ))}
      </svg>
      {bands.map((l, li) => {
        const all = l.rows.flat();
        const solved = all.reduce((s, n) => s + n.solved, 0);
        const total = all.reduce((s, n) => s + n.total, 0);
        return (
          <section key={l.id} className={cn(li > 0 && "mt-6 border-t border-dashed border-white/[0.07] pt-6")}>
            {bands.length > 1 || li === 0 ? (
              <div className="mb-1 flex items-baseline justify-center gap-2 text-center">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{l.title}</span>
                <span className="font-mono text-[10.5px] text-muted-foreground/60">{solved}/{total}</span>
              </div>
            ) : null}
            {l.rows.map((row, ri) => (
              <div key={ri} className="flex flex-wrap justify-center gap-x-5 gap-y-5 py-4 sm:gap-x-7">
                {row.map((n) => (
                  <NodeBox
                    key={n.id}
                    n={n}
                    i={n.index}
                    isNext={n.id === nextModule}
                    selected={n.id === selected}
                    onPick={onPick}
                    onHover={setHover}
                    refFn={(el) => { if (el) boxes.current.set(n.id, el); else boxes.current.delete(n.id); }}
                  />
                ))}
              </div>
            ))}
          </section>
        );
      })}
    </div>
  );
}

export function RoadmapLegend() {
  const item = (cls: string, text: string) => (
    <span className="flex items-center gap-1.5"><span className={cn("h-2.5 w-2.5 rounded-sm border", cls)} />{text}</span>
  );
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
      {item("border-emerald-500/60 bg-emerald-500/20", "done")}
      {item("border-primary/60 bg-primary/20", "in progress")}
      {item("border-white/30 bg-[#161a24]", "ready")}
      {item("border-white/10 bg-[#12151c] opacity-60", "needs earlier modules")}
      <span className="flex items-center gap-1"><RotateCcw className="h-3 w-3 text-sky-300" /> due for review</span>
    </div>
  );
}
