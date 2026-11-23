// Bundle Monaco locally instead of fetching it from a CDN at runtime, so the
// editor works offline. Only Python is registered.
import { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor/esm/vs/editor/editor.api.js";
import "monaco-editor/esm/vs/editor/editor.all.js";
import "monaco-editor/esm/vs/basic-languages/python/python.contribution.js";
import EditorWorker from "monaco-editor/esm/vs/editor/editor.worker.js?worker";

(self as unknown as { MonacoEnvironment: unknown }).MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};

monaco.editor.defineTheme("kata-dark", {
  base: "vs-dark",
  inherit: true,
  rules: [
    { token: "comment", foreground: "6b7280", fontStyle: "italic" },
    { token: "keyword", foreground: "c792ea" },
    { token: "number", foreground: "f78c6c" },
    { token: "string", foreground: "c3e88d" },
  ],
  colors: {
    "editor.background": "#0f1117",
    "editor.lineHighlightBackground": "#1a1d26",
    "editorLineNumber.foreground": "#3b4150",
    "editorLineNumber.activeForeground": "#9aa3b5",
    "editorGutter.background": "#0f1117",
    "editor.selectionBackground": "#2a3350",
    "editorIndentGuide.background1": "#1d212b",
    "editorWidget.background": "#161922",
    "scrollbarSlider.background": "#ffffff14",
  },
});
// Generates the token CSS that monaco.editor.colorize output relies on.
monaco.editor.setTheme("kata-dark");

loader.config({ monaco });

export { monaco };
