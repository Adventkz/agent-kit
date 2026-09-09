export type Trace = {
  name: string;
  arguments: string;
  result?: string;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  traces?: Trace[];
};

export type AgentEvent =
  | { type: "token"; text: string }
  | { type: "tool_call"; name: string; arguments: string }
  | { type: "tool_result"; name: string; result: string }
  | { type: "done" }
  | { type: "error"; message: string };
