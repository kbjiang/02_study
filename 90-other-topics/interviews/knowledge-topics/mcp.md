# How to Make an LLM MCP Capable

To make a Large Language Model (LLM) capable of using the **Model Context Protocol (MCP)**, you do not modify the core weights of the LLM itself. Instead, you build a **client orchestration layer** around the LLM that translates its standard tool-calling capabilities into standard MCP requests.

Because MCP decouples the model from the underlying infrastructure, the LLM simply acts as the central decision-maker. It reads the user request, decides which MCP resource, prompt, or tool it needs, and issues a standard tool call or structured text output. The client framework then routes that choice to the correct MCP server.

---

### The Architecture: How an LLM Uses MCP

```text
┌─────────────┐       Native Tool Call      ┌────────────┐
│             │ ──────────────────────────> │            │
│     LLM     │                             │ MCP Client │
│             │ <────────────────────────── │  (Wrapper) │
└─────────────┘       Tool Result JSON      └────────────┘
                                                  │ ▲
                                  JSON-RPC        │ │  Stdio /
                                  Over Transport  ▼ │  Sse
                                            ┌────────────┐
                                            │ MCP Server │
                                            └────────────┘
```

An LLM becomes "MCP capable" when your system wraps it in an **MCP Client Application**. The process involves three main implementation steps:

1. **Schema Translation**
2. **Transport Connection**
3. **The Execution Loop**
