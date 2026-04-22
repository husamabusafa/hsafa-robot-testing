












I have a complete picture now. Let me write up a concise, accurate explanation of Hsafa Core based on the actual code.

# Hsafa Core

Hsafa Core is **a brain-as-a-service**. It runs at `@/Users/Husam/Dev/hsafa-logic/hsafa-core/core` as an Express HTTP service (`@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/index.ts:1-115`) and its job is to host **Haseefs** — intelligent entities (حصيف = "the wise one") that perceive events, think, remember, and act.

The defining idea (from `@/Users/Husam/Dev/hsafa-logic/hsafa-core/hsafa-docs-versions/v7/v7-hsafa-core.md:5`):

> *Hsafa Core is a brain. External services give it senses and hands via a simple SDK.*

Core does not talk to WhatsApp, email, robots, databases, or chat UIs directly. It never owns API keys or hardware. Instead, any external service connects via `@hsafa/sdk` and plugs in **skills** (groups of tools). Core does the thinking.

## The Four Pillars of a Haseef

Every Haseef is defined by four things (`@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/prisma/schema.prisma:33-51`):

- **Profile** — `profileJson` — who the Haseef *is* (phone, email, robotId, language). This is also the **routing table**.
- **Config** — `configJson` — model, instructions, persona.
- **Skills** — `skills: String[]` — which external services this Haseef is allowed to use (e.g. `["spaces", "whatsapp", "body", "vision"]`).
- **Memory** — four separate tables: `EpisodicMemory`, `SemanticMemory`, `SocialMemory`, `ProceduralMemory`.

## Core Features

### 1. Stateless trigger-based architecture
There are **no living agent processes**. A Haseef doesn't "run" idly. It is *invoked* only when an event arrives. From `@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/index.ts:92-93`:
```
v7: No process startup needed — haseefs are triggered on-demand by events
Ready — waiting for events
```

### 2. Event routing by profile
`@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/lib/event-router.ts:42-90` resolves an incoming event to a Haseef in two ways:
- **Direct** — `event.haseefId`
- **Profile-based** — `event.target: { phone: "+966..." }` → search `profileJson` for a matching field.

This means a WhatsApp service doesn't need to know Haseef IDs. It just says *"a message arrived for this phone"* and Core finds the right brain.

### 3. Coordinator with interrupts
`@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/lib/coordinator.ts:30-61` enforces **one run per Haseef at a time**. If a new event arrives while the Haseef is already thinking, the current run is `AbortController.abort()`ed and a fresh run starts with the new context. This mimics human attention: you drop what you're doing when something new happens.

### 4. The Invoker — the think loop
`@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/lib/invoker.ts:64-302` implements `perceive → think → act → remember`:
1. Load Haseef from DB
2. Create `Run` record
3. **Assemble memory** (all 4 types in parallel) — `@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/memory/selection.ts:30-54`
4. Build system prompt (IDENTITY + PROFILE + MEMORY + INSTRUCTIONS)
5. Build tools: prebuilt (`done`, `set_memories`, `delete_memories`, `recall_memories`) + all tools from the Haseef's active skills
6. `streamText()` with AI SDK v6, `stopWhen: [hasToolCall('done'), stepCountIs(50)]`
7. Stream text deltas + tool events to Redis Pub/Sub for live UIs
8. **Post-run reflection** — `@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/memory/reflection.ts:24-34` stores an episodic memory of what happened.

### 5. Four-type memory system
Unlike typical "one big vector store" agents, Core structures memory the way cognitive science does (`@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/memory/`):
- **Semantic** — facts/knowledge, key-value, importance 1-10
- **Episodic** — past run summaries (what happened)
- **Social** — person models (who you know, relationships, observations)
- **Procedural** — learned `trigger → response` patterns with confidence scores

Each run assembles a personalized slice: top-importance facts + recent episodes + all known people + procedures matching the current event.

### 6. Tool dispatcher via SSE (not Redis Streams)
`@/Users/Husam/Dev/hsafa-logic/hsafa-core/core/src/lib/tool-dispatcher.ts:109-154` is clever: when the LLM calls a tool, Core picks **one** connected SSE client for that skill (unicast, not broadcast) and `await`s a Promise that resolves when the service POSTs back to `/api/actions/:actionId/result`. Timeout = 30s default.

### 7. Universal SDK contract
Services only ever implement three methods:
- `registerTools([...])` → `PUT /api/skills/:skill/tools`
- `onToolCall(name, handler)` → locally registered
- `pushEvent({...})` → `POST /api/events`
- `connect()` → opens SSE to `GET /api/skills/:skill/actions/stream`

A WhatsApp integration, a robot body, and a chat UI all use the exact same 3 methods.

### 8. API-key-scoped ownership
Each API key owns the Haseefs and skills it creates. Platform skills (`apiKeyId = null`, like `spaces`) are shared. This is what lets multi-tenant products like hsafa-spaces safely share one Core.

## Why Hsafa Core Is Special

1. **Brain vs. body separation.** Most agent frameworks tangle reasoning, tool implementations, and integrations in one codebase. Core is *only* the brain. Services run anywhere — on a robot, inside a Next.js app, behind WhatsApp's webhook — and all look identical to the brain.

2. **Profile-as-routing-table.** A Haseef's identity literally *is* its routing config. No separate binding tables, no scope-specific settings. Add a phone number to Atlas's profile and WhatsApp events for that number automatically flow to Atlas.

3. **Interrupt-driven cognition.** A new event preempts the current run. This gives Haseefs a property almost no framework has: they can be *pulled away* mid-thought by something more urgent, just like people.

4. **Structured cognitive memory.** Four distinct memory types with different retrieval strategies, plus automatic post-run reflection. Not just RAG over a blob — a model of *how minds remember*.

5. **Trigger-based, zero idle cost.** No worker pools. No persistent agent processes. Core is idle between events. Scales from one Haseef to thousands on the same box.

6. **LLM as the integration layer.** Because every skill is just a bag of tools + event pusher, combining WhatsApp + robot body + vision + voice requires *zero* integration code — the LLM composes them naturally (`@/Users/Husam/Dev/hsafa-logic/hsafa-core/hsafa-docs-versions/v7/v7-hsafa-core.md:909-928`).

7. **One SDK, infinite body plans.** The same three-method SDK serves a chat server, an email scope, a robot's onboard computer, or a game bot. The Haseef doesn't care — it just has more tools today than yesterday.

In one line: **Core is an LLM-powered event loop with memory, and everything else — your app, your hardware, your integrations — is just a service that speaks a 3-method protocol to it.**