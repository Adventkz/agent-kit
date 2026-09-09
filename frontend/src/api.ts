import type { AgentEvent, Message } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function health() {
  const response = await fetch(`${BASE}/api/health`);
  if (!response.ok) throw new Error(`health ${response.status}`);
  return (await response.json()) as {
    model: string;
    tools: string[];
    documents: string[];
    api_key_configured: boolean;
  };
}

export async function uploadDoc(file: File) {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(`${BASE}/api/docs`, { method: "POST", body });
  if (!response.ok) throw new Error((await response.text()) || `upload ${response.status}`);
  return (await response.json()) as { source: string; chunks: number; documents: string[] };
}

export async function clearDocs() {
  await fetch(`${BASE}/api/docs`, { method: "DELETE" });
}

/** POST + SSE: EventSource cannot send a body, so we parse the stream ourselves. */
export async function streamChat(
  messages: Message[],
  onEvent: (event: AgentEvent) => void,
) {
  const response = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
    }),
  });
  if (!response.body) throw new Error("no response stream");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      onEvent(JSON.parse(line.slice(6)) as AgentEvent);
    }
  }
}
