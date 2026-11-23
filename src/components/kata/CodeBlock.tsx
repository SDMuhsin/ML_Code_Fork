import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";
import { monaco } from "@/lib/monaco";
import { cn } from "@/lib/utils";

const cache = new Map<string, string>();

/** Python syntax highlighting via Monaco's tokenizer, so it matches the editor exactly. */
export function useHighlighted(code: string, language = "python") {
  const key = language + "\u0000" + code;
  const [html, setHtml] = useState<string | null>(cache.get(key) ?? null);
  useEffect(() => {
    if (cache.has(key)) {
      setHtml(cache.get(key)!);
      return;
    }
    let alive = true;
    monaco.editor.colorize(code, language, { tabSize: 4 }).then((h) => {
      cache.set(key, h);
      if (alive) setHtml(h);
    });
    return () => {
      alive = false;
    };
  }, [key, code, language]);
  return html;
}

export function CodeBlock({
  code,
  language = "python",
  className,
  copyable = true,
  maxHeight,
}: {
  code: string;
  language?: string;
  className?: string;
  copyable?: boolean;
  maxHeight?: number;
}) {
  const html = useHighlighted(code, language === "text" ? "plaintext" : language);
  const [copied, setCopied] = useState(false);
  return (
    <div className={cn("group relative rounded-lg border border-white/[0.06] bg-[#0f1117]", className)}>
      {copyable && (
        <button
          onClick={() => {
            navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1200);
          }}
          className="absolute right-2 top-2 rounded-md p-1.5 text-muted-foreground opacity-0 transition-opacity hover:bg-white/5 hover:text-foreground group-hover:opacity-100"
          title="Copy"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      )}
      <pre
        className="overflow-x-auto px-4 py-3 font-mono text-[12.5px] leading-[1.6] text-[#d4d4d4]"
        style={maxHeight ? { maxHeight, overflowY: "auto" } : undefined}
      >
        {html ? <code dangerouslySetInnerHTML={{ __html: html }} /> : <code>{code}</code>}
      </pre>
    </div>
  );
}

/** One-line inline highlighted expression (used in predict items). */
export function InlineCode({ code }: { code: string }) {
  const html = useHighlighted(code);
  return html ? (
    <code className="font-mono text-[12.5px]" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <code className="font-mono text-[12.5px]">{code}</code>
  );
}
