# Hsafa Skill — AI Instructions

> **This file is for AI assistants (Cursor, Windsurf, Copilot, etc.).** Read all `.md` files in this `.hsafa/` folder to understand the Hsafa platform (v7) and how to write skills correctly.

## Context Files

Read these files in order for full context:

1. **`what-is-hsafa.md`** — What Hsafa is, the Core + Services architecture
2. **`sdk-reference.md`** — Pointer to the canonical SDK reference
3. **`cli-reference.md`** — All CLI commands
4. **`skill-development-guide.md`** — Best practices, patterns, anti-patterns
5. **`examples.md`** — Real code examples (API wrapper, database, webhooks, monitoring)

## Rules for AI

When generating code for this Hsafa skill project:

1. **Always use `@hsafa/sdk`** — `import { HsafaSDK } from "@hsafa/sdk"`.
2. **Constructor uses `skill`, not `scope`** — `new HsafaSDK({ coreUrl, apiKey, skill })`.
3. **Authenticate with the single Core key** — env var `HSAFA_CORE_KEY` (Core's `SECRET_KEY`). There is no per-skill key in v7.
4. **Use `snake_case` for tool names** — e.g. `get_weather`, `send_email`.
5. **Add descriptions to every tool and every input field** — the haseef reads them to choose tools.
6. **Return structured JSON from handlers** — not strings, not raw HTML.
7. **Use `hsafa.memory.*`, `hsafa.haseef.*`, `hsafa.runs.*`** for state beyond tool calls.
8. **Handle errors gracefully** — return `{ error: "message" }` or throw.
9. **Include graceful shutdown** — `hsafa.disconnect()` on `SIGINT` / `SIGTERM`.
10. **Keep tools focused** — one tool = one action; split complex workflows into multiple tools.

## This Project

This is a Hsafa skill service. It connects to Hsafa Core and provides tools to haseefs (autonomous AI agents). The haseef decides when to call tools — your job is to define what tools are available and implement the handler logic.
