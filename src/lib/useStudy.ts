import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { fmtDue, kataApi, label, type StudyNext } from "@/lib/kata";

/** Everything that shows progress or the queue; call after any run, grade or review change. */
export function invalidateStudy(qc: QueryClient) {
  qc.invalidateQueries({ queryKey: ["kata-catalog"] });
  qc.invalidateQueries({ queryKey: ["kata-study"] });
  qc.invalidateQueries({ queryKey: ["kata-study-next"] });
}

/** Next within one track. Waits until the track is known (the problem has loaded). */
export function useStudyNext(track?: string) {
  return useQuery({
    queryKey: ["kata-study-next", track],
    queryFn: () => kataApi.studyNext(track),
    enabled: Boolean(track),
    refetchInterval: 60_000,
  });
}

/** The words on a Next button: "Review 1c.2", "Start 1d.1", "Continue 1d.1". */
export function describeNext(n?: StudyNext | null) {
  if (!n) return { verb: "Next", id: "", title: "", sub: "" };
  if (n.kind === "done") {
    return { verb: "All caught up", id: "", title: "", sub: n.nextDue ? `next review ${fmtDue(n.nextDue)}` : "every problem is solved" };
  }
  const id = n.id ? label(n.id) : "";
  if (n.kind === "review") {
    const more = (n.dueCount ?? 1) > 1 ? ` · ${n.dueCount} due` : "";
    return { verb: n.resumed ? "Resume review" : "Review", id, title: n.title ?? "", sub: `due for review${more}` };
  }
  return {
    verb: n.started ? "Continue" : "Start",
    id,
    title: n.title ?? "",
    sub: n.skippedAhead ? "its prerequisites are not finished yet" : `next on the path · ${n.moduleTitle ?? ""}`,
  };
}

/** Go to whatever is next: starts the review if it is one, then opens the problem. */
export function useGoNext() {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const go = useCallback(async (given?: StudyNext | null, track?: string) => {
    setBusy(true);
    try {
      const n = given ?? (await kataApi.studyNext(track));
      if (n.kind === "done" || !n.id) {
        toast.success("All caught up", { description: n.nextDue ? `The next review is due ${fmtDue(n.nextDue)}.` : "Every problem is solved." });
        return;
      }
      if (n.kind === "review") {
        const r = await kataApi.startReview(n.id);
        qc.setQueryData(["kata-problem", n.id], r.problem);
      }
      invalidateStudy(qc);
      navigate(`/kata/${n.id}`);
    } catch (e: any) {
      toast.error("Could not open the next problem", { description: e.message });
    } finally {
      setBusy(false);
    }
  }, [qc, navigate]);
  return { go, busy };
}
