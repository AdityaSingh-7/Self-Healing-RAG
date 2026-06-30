import { getSession } from "next-auth/react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

/**
 * Fetch wrapper that attaches the auth token to all backend requests.
 */
async function fetchWithAuth(path: string, options: RequestInit = {}) {
  const session = await getSession();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  // Attach JWT token if logged in
  if (session) {
    headers["Authorization"] = `Bearer ${(session as any).accessToken || ""}`;
  }

  const response = await fetch(`${BACKEND_URL}${path}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response;
}

/**
 * Upload a PDF document.
 */
export async function uploadDocument(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetchWithAuth("/ingest/upload", {
    method: "POST",
    body: formData,
  });

  return response.json();
}

/**
 * Ask a question (non-streaming).
 */
export async function queryRAG(question: string, options?: { topK?: number; recencyWeight?: number }) {
  const response = await fetchWithAuth("/query/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      top_k: options?.topK || 5,
      recency_weight: options?.recencyWeight || 0.2,
      stream: false,
    }),
  });

  return response.json();
}

/**
 * Ask a question with SSE streaming.
 * Returns a ReadableStream that yields server-sent events.
 */
export async function queryRAGStream(
  question: string,
  onToken: (token: string) => void,
  onSources: (sources: any[]) => void,
  onDone: () => void,
  onError: (error: string) => void,
) {
  try {
    const response = await fetch(`${BACKEND_URL}/query/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: 5,
        recency_weight: 0.2,
        stream: true,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: "Request failed" }));
      onError(err.detail);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      onError("No response body");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") {
            onDone();
            return;
          }

          try {
            const parsed = JSON.parse(data);

            if (parsed.type === "token") {
              onToken(parsed.content);
            } else if (parsed.type === "sources") {
              onSources(parsed.content);
            } else if (parsed.type === "done") {
              onDone();
              return;
            }
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }

    onDone();
  } catch (err: any) {
    onError(err.message || "Stream failed");
  }
}

/**
 * Get list of uploaded documents.
 */
export async function listDocuments() {
  const response = await fetchWithAuth("/documents/");
  return response.json();
}

/**
 * Delete a document.
 */
export async function deleteDocument(docId: string) {
  const response = await fetchWithAuth(`/documents/${docId}`, {
    method: "DELETE",
  });
  return response.json();
}

/**
 * Clear conversation history.
 */
export async function clearHistory() {
  const response = await fetchWithAuth("/query/clear-history", {
    method: "POST",
  });
  return response.json();
}
