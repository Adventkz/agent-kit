import { useEffect, useRef, useState } from "react";
import { clearDocs, health, streamChat, uploadDoc } from "./api";
import type { Message, Trace } from "./types";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [docs, setDocs] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    health()
      .then((h) => {
        setDocs(h.documents);
        setStatus(
          h.api_key_configured
            ? `${h.model} · tools: ${h.tools.join(", ")}`
            : "OPENAI_API_KEY is not set on the backend",
        );
      })
      .catch(() => setStatus("backend unreachable"));
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;

    const history: Message[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "", traces: [] }]);
    setInput("");
    setBusy(true);

    const patchLast = (fn: (m: Message) => Message) =>
      setMessages((prev) => prev.map((m, i) => (i === prev.length - 1 ? fn(m) : m)));

    try {
      await streamChat(history, (event) => {
        if (event.type === "token") {
          patchLast((m) => ({ ...m, content: m.content + event.text }));
        } else if (event.type === "tool_call") {
          const trace: Trace = { name: event.name, arguments: event.arguments };
          patchLast((m) => ({ ...m, traces: [...(m.traces ?? []), trace] }));
        } else if (event.type === "tool_result") {
          patchLast((m) => {
            const traces = [...(m.traces ?? [])];
            for (let i = traces.length - 1; i >= 0; i--) {
              if (traces[i].name === event.name && traces[i].result === undefined) {
                traces[i] = { ...traces[i], result: event.result };
                break;
              }
            }
            return { ...m, traces };
          });
        } else if (event.type === "error") {
          patchLast((m) => ({ ...m, content: m.content + `\n\n⚠️ ${event.message}` }));
        }
      });
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setStatus(`uploading ${file.name}…`);
    try {
      const result = await uploadDoc(file);
      setDocs(result.documents);
      setStatus(`${file.name}: ${result.chunks} chunks indexed`);
    } catch (error) {
      setStatus(String(error));
    } finally {
      event.target.value = "";
    }
  }

  return (
    <div className="app">
      <header>
        <h1>agent-kit</h1>
        <span className="status">{status}</span>
      </header>

      <aside>
        <label className="upload">
          <input type="file" accept=".txt,.md,.json,.csv,.pdf" onChange={onUpload} />
          Upload a document
        </label>
        <ul>
          {docs.map((d) => (
            <li key={d}>{d}</li>
          ))}
        </ul>
        {docs.length > 0 && (
          <button
            className="ghost"
            onClick={() => clearDocs().then(() => setDocs([]))}
          >
            Clear index
          </button>
        )}
      </aside>

      <main>
        {messages.length === 0 && (
          <p className="empty">
            Ask something, or upload a document and ask about its contents.
          </p>
        )}
        {messages.map((message, index) => (
          <article key={index} className={message.role}>
            {message.traces?.map((trace, i) => (
              <details key={i} className="trace">
                <summary>
                  🔧 {trace.name}({trace.arguments})
                </summary>
                <pre>{trace.result ?? "…"}</pre>
              </details>
            ))}
            <div className="bubble">{message.content || (busy ? "…" : "")}</div>
          </article>
        ))}
        <div ref={bottom} />
      </main>

      <footer>
        <textarea
          value={input}
          placeholder="Сұрағыңызды жазыңыз / Введите вопрос / Ask a question"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        <button onClick={() => void send()} disabled={busy}>
          {busy ? "…" : "Send"}
        </button>
      </footer>
    </div>
  );
}
