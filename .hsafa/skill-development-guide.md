# Skill Development Guide (v7)

> How to build a high-quality Hsafa skill — best practices, patterns, anti-patterns.

## Project Structure

```
my-skill/
├── .hsafa/                # AI context (this folder)
├── src/
│   ├── index.ts           # SDK setup, register tools, connect
│   ├── tools.ts           # Tool definitions (name, schema, description)
│   └── handler.ts         # Tool call handlers (your logic)
├── .env                   # SKILL_NAME, HSAFA_CORE_URL, HSAFA_CORE_KEY
├── package.json
└── README.md
```

## The 4-Step Pattern

```typescript
import { HsafaSDK } from "@hsafa/sdk";

// 1. CREATE SDK INSTANCE
const hsafa = new HsafaSDK({
  coreUrl: process.env.HSAFA_CORE_URL!,
  apiKey:  process.env.HSAFA_CORE_KEY!,
  skill:   process.env.SKILL_NAME!,
});

// 2. REGISTER TOOLS
await hsafa.registerTools(tools);

// 3. HANDLE TOOL CALLS
hsafa.onToolCall("tool_name", async (args, ctx) => {
  return { success: true };
});

// 4. CONNECT
hsafa.connect();
```

## Writing Good Tools

### Naming
- Use `snake_case`: `get_weather`, `send_email`, `list_tables`
- Be specific: `search_customers` not `search`
- Verb-prefix: `get_`, `list_`, `create_`, `update_`, `delete_`, `send_`, `run_`

### Descriptions
The haseef reads the description to decide when to use a tool.

```typescript
// GOOD
{ description: "Run a read-only SQL query (SELECT only). Returns rows as JSON. LIMIT enforced automatically." }

// BAD
{ description: "Query the database." }
```

### Input Schemas
Add `description` to every field. Prefer the shorthand for simple types:

```typescript
{
  name: "get_weather",
  description: "Get current weather for a city",
  input: { city: "string", units: "string?" },  // ? = optional
}
```

For complex inputs use raw JSON Schema via `inputSchema`.

## Reading / Writing Memory

A handler always has `ctx.haseef.id` — use it with the memory namespace:

```typescript
hsafa.onToolCall("remember_preference", async (args, ctx) => {
  await hsafa.memory.set(ctx.haseef.id, [
    { key: "preferred_units", value: args.units, importance: 6 },
  ]);
  return { saved: true };
});

hsafa.onToolCall("recall_preferences", async (args, ctx) => {
  const facts = await hsafa.memory.search(ctx.haseef.id, "preference");
  return { facts };
});
```

The 4 memory types:

| Namespace call | Type | Use it for |
|---|---|---|
| `memory.list / search / set / delete` | semantic | key/value facts the haseef should remember |
| `memory.episodes / searchEpisodes` | episodic | summaries of past runs |
| `memory.social` | social | what the haseef knows about specific people |
| `memory.procedural` | procedural | learned patterns / "how to" knowledge |

## Pushing Events

Use `pushEvent` when something external happens that should reach a haseef.

```typescript
// Direct routing by haseef ID:
await hsafa.pushEvent({
  type: "new_order",
  data: { orderId, total },
  haseefId: targetHaseefId,
});

// Or route by profile field — Core finds the matching haseef:
await hsafa.pushEvent({
  type: "whatsapp_message",
  data: { text },
  target: { phone: "+15555551234" },
});
```

The skill name is added automatically — don't pass it.

## Handler Best Practices

### Return structured data
```typescript
// GOOD
return { customers: [...], totalCount: 42, hasMore: true };
// BAD
return "Found 42 customers";
```

### Handle errors gracefully
```typescript
hsafa.onToolCall("query", async (args) => {
  try {
    return { rows: await db.query(args.sql) };
  } catch (err) {
    return { error: (err as Error).message, hint: "Check your SQL syntax" };
  }
});
```

### Keep handlers focused
One tool = one action. Split complex workflows into multiple tools.

## Anti-Patterns

- ❌ Tools too broad — one tool = one action.
- ❌ Returning raw HTML or huge blobs — return structured JSON.
- ❌ Holding state between tool calls — each call is independent. Use memory for state.
- ❌ Generic tool names — `run`, `do`, `action` tell the haseef nothing.
- ❌ Skipping graceful shutdown — call `hsafa.disconnect()` on `SIGINT` / `SIGTERM`.
- ❌ Hardcoding Core's URL or key — read `HSAFA_CORE_URL` and `HSAFA_CORE_KEY`.
