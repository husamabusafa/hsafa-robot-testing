# What is Hsafa (v7)

> Context for AI assistants generating code in this Hsafa skill project.

## Overview

**Hsafa** is a runtime for autonomous AI agents called **haseefs**. A haseef is not a chatbot — it is a long-lived agent with:

- **Identity** — name, description, profile (phone, email, robotId, …)
- **4 memory types** — semantic (key/value facts), episodic (run summaries), social (people), procedural (learned patterns)
- **Skills** — named groups of tools, registered by services
- **Trigger-based execution** — events trigger runs; no continuous loop, no consciousness

## Architecture: Core + Services

Hsafa follows a strict **Core + Services** separation.

### Hsafa Core
The agent's **brain**. Stateless trigger-driven runs (`coordinator → invoker → reflect`). Owns the haseef profile, the 4 memory types, the skill/tool registry, and event routing.

- **API**: REST + SSE at `http://localhost:3001` (default)
- **Auth**: a single shared `SECRET_KEY` sent as `x-api-key` (or `?api_key=` for SSE)
- **Stateless**: every run builds a fresh prompt; there is no persistent conversation log

### Services (one per skill)
Independent processes that connect to Core and provide tools to haseefs. Each service registers under a **skill** — a named channel.

Examples: `spaces`, `scheduler`, `whatsapp`, `postgres`, `weather`. One service per skill — if you need two WhatsApp providers, use `whatsapp_twilio` and `whatsapp_meta`.

A service does four things via `@hsafa/sdk`:

1. `registerTools([...])` — declare tools to Core
2. `onToolCall(name, handler)` — handle tool calls when they arrive
3. `pushEvent({...})` — push outside-world events into haseefs
4. `connect()` — open the long-lived SSE stream

The SDK also exposes `hsafa.memory.*`, `hsafa.haseef.*`, and `hsafa.runs.*` for everything else a skill might need.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Haseef** | A long-lived AI agent with identity, memory, and skills |
| **Skill** | A named channel identifying a service (e.g. `weather`, `spaces`) |
| **Tool** | An action a haseef can take, defined and executed by a skill's service |
| **Event** | Outside-world signal pushed into a haseef; triggers a run if one isn't already running |
| **Run** | A single trigger → think → act → done execution |
| **Memory (4 types)** | semantic / episodic / social / procedural — read/write via `hsafa.memory.*` |

## End-to-end Flow

```
1. Skill service starts → connects to Core via @hsafa/sdk
2. Registers tools → Core stores them under the skill name
3. Skill is added to a haseef's skills[] array → haseef can use the tools
4. External event happens → service calls hsafa.pushEvent({ haseefId or target, … })
5. Core triggers a run → builds prompt → LLM picks tools → tool calls fan out via SSE
6. Service handler runs → returns result → Core continues thinking → run finishes
```

## This Project is a Skill

This project is a Hsafa skill — a service that connects to Hsafa Core via `@hsafa/sdk`. When generating code, follow the patterns in `examples.md` and the rules in `instructions.md`.
