# Hsafa Skill Examples (v7)

> Code examples for common skill patterns. All use `@hsafa/sdk`.

## REST API Wrapper

```typescript
import { HsafaSDK } from "@hsafa/sdk";

const hsafa = new HsafaSDK({
  coreUrl: process.env.HSAFA_CORE_URL!,
  apiKey:  process.env.HSAFA_CORE_KEY!,
  skill:   process.env.SKILL_NAME!,
});

await hsafa.registerTools([
  {
    name: "get_weather",
    description: "Get current weather for a city. Returns temperature, conditions, humidity.",
    inputSchema: {
      type: "object",
      properties: {
        city:  { type: "string", description: "City name (e.g. \"Tokyo\")" },
        units: { type: "string", enum: ["metric", "imperial"], description: "Temperature units" },
      },
      required: ["city"],
    },
  },
]);

hsafa.onToolCall("get_weather", async (args) => {
  const API_KEY = process.env.WEATHER_API_KEY!;
  const units   = (args.units as string) || "metric";
  const res = await fetch(
    `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(args.city as string)}&units=${units}&appid=${API_KEY}`
  );
  if (!res.ok) return { error: `City "${args.city}" not found` };
  const data = await res.json();
  return {
    city:        data.name,
    temperature: data.main.temp,
    conditions:  data.weather[0].description,
    humidity:    data.main.humidity,
  };
});

hsafa.connect();
```

## Database Skill (with memory)

```typescript
import { HsafaSDK } from "@hsafa/sdk";
import pg from "pg";

const hsafa = new HsafaSDK({
  coreUrl: process.env.HSAFA_CORE_URL!,
  apiKey:  process.env.HSAFA_CORE_KEY!,
  skill:   process.env.SKILL_NAME!,
});

const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

await hsafa.registerTools([
  {
    name: "query",
    description: "Run a read-only SQL query (SELECT only). Returns rows as JSON.",
    inputSchema: {
      type: "object",
      properties: { sql: { type: "string", description: "SELECT query" } },
      required: ["sql"],
    },
  },
  {
    name: "list_tables",
    description: "List all tables in the database.",
    input: {},
  },
]);

hsafa.onToolCall("query", async (args, ctx) => {
  const sql = (args.sql as string).trim();
  if (!sql.toUpperCase().startsWith("SELECT")) {
    return { error: "Only SELECT queries are allowed" };
  }
  const result = await pool.query(sql);

  // Remember the last query for this haseef
  await hsafa.memory.set(ctx.haseef.id, [
    { key: "last_query", value: sql, importance: 4 },
  ]);

  return { rows: result.rows, rowCount: result.rowCount };
});

hsafa.onToolCall("list_tables", async () => {
  const result = await pool.query(
    "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
  );
  return { tables: result.rows.map((r) => r.tablename) };
});

hsafa.connect();

process.on("SIGINT", async () => {
  hsafa.disconnect();
  await pool.end();
  process.exit(0);
});
```

## Webhook Listener + Push Events

```typescript
import { HsafaSDK } from "@hsafa/sdk";
import express from "express";

const hsafa = new HsafaSDK({
  coreUrl: process.env.HSAFA_CORE_URL!,
  apiKey:  process.env.HSAFA_CORE_KEY!,
  skill:   process.env.SKILL_NAME!,
});

await hsafa.registerTools([
  {
    name: "list_events",
    description: "List recent webhook events.",
    input: { limit: "number?" },
  },
]);

const events: Array<{ type: string; data: unknown; receivedAt: string }> = [];

hsafa.onToolCall("list_events", async (args) => {
  const limit = (args.limit as number) || 10;
  return { events: events.slice(-limit) };
});

hsafa.connect();

const app = express();
app.use(express.json());

app.post("/webhook", async (req, res) => {
  const event = { type: req.body.type ?? "unknown", data: req.body, receivedAt: new Date().toISOString() };
  events.push(event);

  // Forward to a haseef as a sense event — route by phone in this example
  await hsafa.pushEvent({
    type:   `webhook_${event.type}`,
    data:   event.data as Record<string, unknown>,
    target: { phone: req.body.phone },
  }).catch((err) => console.error("Push failed:", err));

  res.json({ received: true });
});

app.listen(3100);
```

## Monitoring + Alerts

```typescript
import { HsafaSDK } from "@hsafa/sdk";

const hsafa = new HsafaSDK({
  coreUrl: process.env.HSAFA_CORE_URL!,
  apiKey:  process.env.HSAFA_CORE_KEY!,
  skill:   "monitoring",
});

await hsafa.registerTools([
  {
    name: "get_system_status",
    description: "Get current system health metrics (CPU, memory, disk).",
    input: {},
  },
]);

hsafa.onToolCall("get_system_status", async () => ({
  cpu:    { usage: 45 },
  memory: { percent: 38 },
  disk:   { percent: 36 },
}));

hsafa.connect();

// Poll and push alerts to all haseefs that have this skill
setInterval(async () => {
  const cpu = await getCpuUsage();
  if (cpu < 80) return;

  const haseefs = await hsafa.haseef.list();
  for (const h of haseefs) {
    if (!h.skills?.includes("monitoring")) continue;
    await hsafa.pushEvent({
      type: "cpu_alert",
      data: { severity: cpu > 95 ? "critical" : "warning", cpuUsage: cpu },
      haseefId: h.id,
    });
  }
}, 60_000);

declare function getCpuUsage(): Promise<number>;
```

## Common Patterns

### Retry with backoff
```typescript
async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try { return await fn(); }
    catch (err) {
      if (i === maxRetries - 1) throw err;
      await new Promise((r) => setTimeout(r, 1000 * 2 ** i));
    }
  }
  throw new Error("unreachable");
}
```

### Event logging
```typescript
hsafa.on("run.started",   (e) => console.log(`[${e.haseef.name}] run started`));
hsafa.on("tool.error",    (e) => console.error(`[${e.toolName}] ${e.error}`));
hsafa.on("run.completed", (e) => console.log(`run done in ${e.durationMs}ms`));
```
