# @hsafa/sdk Reference

The full, authoritative SDK reference lives in the SDK package itself:

→ **[`sdks/hsafa-sdk/README.md`](https://www.npmjs.com/package/@hsafa/sdk)** (or in the monorepo: `sdks/hsafa-sdk/README.md`)

It covers:

- The 4 core methods: `registerTools`, `onToolCall`, `pushEvent`, `connect` / `on`
- The `hsafa.memory.*` namespace — read/write all 4 memory types
- The `hsafa.haseef.*` namespace — CRUD haseefs, profile, skills
- The `hsafa.runs.*` namespace — list/get past runs
- All exported types

This file is intentionally a pointer so there is exactly one source of truth for the SDK API.

## TL;DR

```typescript
import { HsafaSDK } from "@hsafa/sdk";

const hsafa = new HsafaSDK({
  coreUrl: process.env.HSAFA_CORE_URL!,  // "http://localhost:3001"
  apiKey:  process.env.HSAFA_CORE_KEY!,  // Core's SECRET_KEY
  skill:   process.env.SKILL_NAME!,      // unique skill name
});

await hsafa.registerTools([
  { name: "ping", description: "Reply with pong", input: {} },
]);

hsafa.onToolCall("ping", async (args, ctx) => {
  // ctx.haseef = { id, name, profile }
  // ctx.actionId = unique action ID
  return { pong: true };
});

hsafa.connect(); // SSE stream, auto-reconnects 2s → 30s
```

## Reading haseef memory inside a handler

```typescript
hsafa.onToolCall("summarize_my_day", async (args, ctx) => {
  const recent = await hsafa.memory.search(ctx.haseef.id, "today", 10);
  return { summary: recent.map(m => m.value).join("\n") };
});
```
