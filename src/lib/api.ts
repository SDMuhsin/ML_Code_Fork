/** Base URL of the MLCode API server. Set VITE_API_URL in .env to override. */
export const API_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "http://localhost:5001";
