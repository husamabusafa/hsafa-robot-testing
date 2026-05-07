# Hsafa Core v7 — Architecture Specification

## One Sentence

Hsafa Core is a brain. External services give it senses and hands via a simple SDK.

---

## Philosophy

A Haseef is built from four pillars:

- **Profile** — who the Haseef IS (identity data, managed by admin)
- **Memory** — what the Haseef KNOWS (learned knowledge, managed by the Haseef itself)
- **Tools** — what the Haseef can DO (registered by external services via SDK)
- **Senses** — what's HAPPENING (events pushed by external services via SDK)

Core is the brain. It thinks, remembers, and decides. External services are the body. They perceive the world and execute actions. Services connect through a universal SDK with three methods: `registerTools`, `onToolCall`, `pushEvent`.

No plugins. No abilities. No in-process extensions. Services run wherever they want — separate servers, the same machine, a robot's onboard computer — and connect to Core over HTTP.

```
┌──────────────────────┐         ┌──────────────────────┐
│  Hsafa Core           │         │  Any Service          │
│                       │   SDK   │                       │
│  LLM (thinks)         │◄───────►│  hsafa.registerTools()│
│  Memory (remembers)   │         │  hsafa.onToolCall()   │
│  Coordinator (routes) │         │  hsafa.pushEvent()    │
│  Profile (identity)   │         │                       │
│  Dashboard (manage)   │         │  Owns its own logic   │
│                       │         │  Owns its own secrets  │
└──────────────────────┘         └──────────────────────┘
```

---

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │         DASHBOARD            │
                    │  Create haseefs, toggle      │
                    │  scopes, edit profiles,      │
                    │  browse memory, view runs    │
                    └──────────────┬──────────────┘
                                   │ HTTP API
                    ┌──────────────▼──────────────┐
                    │        HSAFA CORE            │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │     COORDINATOR        │  │
                    │  │  trigger → invoke      │  │
                    │  │  interrupt if busy     │  │
                    │  └───────────┬────────────┘  │
                    │              │                │
                    │  ┌───────────▼────────────┐  │
                    │  │     INVOKER            │  │
                    │  │  load profile          │  │
                    │  │  assemble memory       │  │
                    │  │  build prompt + tools  │  │
                    │  │  streamText (LLM)      │  │
                    │  │  post-run reflection   │  │
                    │  └───────────┬────────────┘  │
                    │              │                │
                    │  ┌───────────▼────────────┐  │
                    │  │   TOOL REGISTRY        │  │
                    │  │   scope → tools        │  │
                    │  │   scope → SSE channel  │  │
                    │  └───────────┬────────────┘  │
                    │              │                │
                    │  ┌───────────▼────────────┐  │
                    │  │   MEMORY SYSTEM        │  │
                    │  │   episodic + semantic   │  │
                    │  │   social + procedural   │  │
                    │  │   pgvector search       │  │
                    │  └────────────────────────┘  │
                    │                              │
                    └───┬──────┬──────┬──────┬─────┘
                        │      │      │      │
                   SSE+HTTP  SSE+HTTP  ...   ...
                        │      │
              ┌─────────▼──┐ ┌─▼──────────┐
              │ WhatsApp   │ │ Spaces     │
              │ Service    │ │ Server     │
              │(@hsafa/sdk)│ │(@hsafa/sdk)│
              └────────────┘ └────────────┘
```

---

## The SDK

### Installation

```bash
npm install @hsafa/sdk
```

### Three Methods

Every service, regardless of what it does, uses the same three methods:

```typescript
import { HsafaSDK } from '@hsafa/sdk';

const hsafa = new HsafaSDK({
  coreUrl: process.env.HSAFA_CORE_URL,   // "https://core.example.com"
  apiKey: process.env.HSAFA_API_KEY,      // "sk-..."
  scope: 'whatsapp',                      // unique name for this service
});

// 1. Tell Core what you can do
await hsafa.registerTools([...]);

// 2. Do the work when asked
hsafa.onToolCall('tool_name', async (args, ctx) => { ... });

// 3. Tell Core what's happening
hsafa.pushEvent({ type: '...', data: {...}, target: { phone: '...' } });

// Start listening for tool calls
hsafa.connect();
```

### What the SDK Does Internally

The developer never sees HTTP or SSE. The SDK handles everything:

| Method | SDK translates to |
|--------|-------------------|
| `registerTools([...])` | `PUT /api/scopes/{scope}/tools` |
| `pushEvent({...})` | `POST /api/events` |
| `onToolCall(name, handler)` | Registers handler locally |
| `connect()` | Opens SSE to `GET /api/scopes/{scope}/actions/stream` |

When Core needs a tool executed:
1. Core sends a tool call request over the SSE connection
2. SDK routes it to the registered handler
3. Handler runs and returns a result
4. SDK posts the result to `POST /api/actions/{actionId}/result`

```typescript
// Inside the SDK — the developer never sees this
class HsafaSDK {
  private handlers = new Map<string, ToolHandler>();

  async registerTools(tools: ToolDefinition[]): Promise<void> {
    await fetch(`${this.coreUrl}/api/scopes/${this.scope}/tools`, {
      method: 'PUT',
      headers: { 'x-api-key': this.apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({ tools }),
    });
  }

  onToolCall(name: string, handler: ToolHandler): void {
    this.handlers.set(name, handler);
  }

  async pushEvent(event: EventPayload): Promise<void> {
    await fetch(`${this.coreUrl}/api/events`, {
      method: 'POST',
      headers: { 'x-api-key': this.apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({ scope: this.scope, ...event }),
    });
  }

  async connect(): Promise<void> {
    const stream = new EventSource(
      `${this.coreUrl}/api/scopes/${this.scope}/actions/stream`,
      { headers: { 'x-api-key': this.apiKey } }
    );

    stream.onmessage = async (event) => {
      const action = JSON.parse(event.data);
      const handler = this.handlers.get(action.toolName);
      if (!handler) return;

      try {
        const result = await handler(action.args, {
          haseef: action.haseef,  // { id, name, profile }
        });
        await fetch(`${this.coreUrl}/api/actions/${action.actionId}/result`, {
          method: 'POST',
          headers: { 'x-api-key': this.apiKey, 'Content-Type': 'application/json' },
          body: JSON.stringify({ result }),
        });
      } catch (err) {
        await fetch(`${this.coreUrl}/api/actions/${action.actionId}/result`, {
          method: 'POST',
          headers: { 'x-api-key': this.apiKey, 'Content-Type': 'application/json' },
          body: JSON.stringify({ error: err.message }),
        });
      }
    };
  }
}
```

### Tool Call Context

When a tool handler runs, it receives the haseef's identity:

```typescript
hsafa.onToolCall('send_message', async (args, ctx) => {
  ctx.haseef.id          // "atlas-uuid"
  ctx.haseef.name        // "Atlas"
  ctx.haseef.profile     // { phone: "+966...", email: "...", robotId: "..." }
});
```

The service uses `ctx.haseef.profile` to know which identity to act as. For example, a WhatsApp service reads `ctx.haseef.profile.phone` to know which number to send from.

---

## Event Routing

When a service pushes an event, Core needs to know which haseef it's for. Two routing modes:

### Mode 1: Route by Profile Field (target)

The service doesn't know haseef IDs. It provides an identifier, and Core matches it against haseef profiles.

```typescript
// "A message arrived for this phone number — Core, figure out who that is"
hsafa.pushEvent({
  type: 'message',
  data: { from: '+966559876543', text: 'Hello!' },
  target: { phone: '+966501234567' },
});
```

Core searches: which haseef has `profile.phone === "+966501234567"`? Found Atlas → trigger Atlas.

### Mode 2: Route by Haseef ID (direct)

The service already knows which haseef to notify.

```typescript
// "This specific haseef needs to know about this"
hsafa.pushEvent({
  type: 'message',
  data: { spaceId: 'space-1', text: 'Hello!' },
  haseefId: 'atlas-uuid',
});
```

### How Core Resolves Events

```typescript
// Inside Core
async function handleEvent(event: IncomingEvent): Promise<void> {
  let haseefId: string;

  if (event.haseefId) {
    // Direct routing
    haseefId = event.haseefId;
  } else if (event.target) {
    // Profile-based routing
    const haseef = await resolveByProfile(event.target);
    if (!haseef) throw new Error('No haseef matches this target');
    haseefId = haseef.id;
  } else {
    throw new Error('Event must have either haseefId or target');
  }

  // Check scope is active for this haseef
  const haseef = await db.haseef.findUnique({ where: { id: haseefId } });
  if (!haseef.scopes.includes(event.scope)) {
    throw new Error(`Scope "${event.scope}" is not active for this haseef`);
  }

  // Trigger the haseef
  coordinator.trigger(haseefId, {
    scope: event.scope,
    type: event.type,
    data: event.data,
    attachments: event.attachments,
  });
}

async function resolveByProfile(target: Record<string, string>): Promise<Haseef | null> {
  for (const [key, value] of Object.entries(target)) {
    const haseef = await db.haseef.findFirst({
      where: {
        profileJson: { path: [key], equals: value }
      }
    });
    if (haseef) return haseef;
  }
  return null;
}
```

### Which Mode to Use

| Scope | Routing mode | Why |
|-------|-------------|-----|
| WhatsApp | `target: { phone }` | Message arrives at a phone number, service doesn't track haseefs |
| Email | `target: { email }` | Email arrives at an address |
| Robot | `target: { robotId }` | Sensor data from a specific robot |
| Game | `target: { playerId }` | Game event for a specific player |
| Spaces | `haseefId` directly | Space server manages membership, knows the IDs |
| Database | `haseefId` directly | DB trigger knows which haseef to notify |
| Company internal | `haseefId` directly | Internal system knows the assignment |

---

## Haseef Profile

Every haseef has a profile — key-value identity data that services use for routing and acting on behalf of the haseef.

```json
{
  "name": "Atlas",
  "profile": {
    "phone": "+966501234567",
    "email": "atlas@company.com",
    "robotId": "reachy-01",
    "playerId": "bot-001",
    "location": "Riyadh",
    "language": "ar"
  },
  "scopes": ["whatsapp", "spaces", "body", "vision", "voice"]
}
```

Profile fields are freeform. Any service can use any field. Core doesn't validate profile field names — it just matches them when routing events.

**The profile is the routing table.** No binding config. No scope-specific config. The haseef's identity IS the config.

---

## Scope Management

### What a Scope Is

A scope is a named group of tools registered by a service. Each service registers under one scope.

**Important constraint: one service per scope.** A scope has exactly one SSE connection, one set of tools, and one service handling tool calls. If you need two WhatsApp providers, use different scopes: `whatsapp_twilio`, `whatsapp_meta`. This keeps dispatch simple — Core never has to decide which of multiple services should handle a tool call.

```
scope: "whatsapp"  → tools: [send_message, send_image, get_contacts]
scope: "email"     → tools: [send_email, get_inbox, reply_email]
scope: "body"      → tools: [move, grab, wave, look_at]
scope: "vision"    → tools: [capture_image, detect_objects, find_person]
```

### Activating Scopes Per Haseef

Each haseef has a list of active scopes. Just a list of strings — toggle on/off.

```json
{
  "name": "Atlas",
  "scopes": ["whatsapp", "spaces", "body", "vision", "voice"]
}
```

```json
{
  "name": "Support",
  "scopes": ["email", "database"]
}
```

When the LLM runs, Core only loads tools from the haseef's active scopes. Atlas gets WhatsApp + Spaces + robot tools. Support gets email + database tools.

### Dashboard: Just Checkboxes

```
┌──────────────────────────────────────────┐
│  Edit: Atlas                             │
│                                          │
│  Profile:                                │
│    Phone:    [+966501234567______]        │
│    Email:    [atlas@company.com__]        │
│    Robot ID: [reachy-01__________]        │
│    [+ Add field]                         │
│                                          │
│  Active Services:                        │
│    ☑ whatsapp    🟢 connected            │
│    ☑ spaces      🟢 connected            │
│    ☑ body        🟢 connected            │
│    ☑ vision      🟢 connected            │
│    ☑ voice       🟢 connected            │
│    ☐ email       🟢 connected            │
│    ☐ database    🟢 connected            │
│    ☐ game        🔴 disconnected         │
│                                          │
│  [Save]                                  │
└──────────────────────────────────────────┘
```

No config forms. No API tokens. Just toggle which services this haseef can use. The service owns its own credentials and logic.

### Who Owns What

```
Core owns:                    Service owns:
  ├─ Haseef profile             ├─ API keys & credentials (env vars)
  ├─ Active scopes list         ├─ Tool implementation logic
  ├─ Event routing              ├─ External connections (APIs, hardware)
  ├─ Tool dispatch to services  ├─ Event detection & pushing
  ├─ Memory system              └─ Its own deployment & scaling
  └─ LLM orchestration
```

### API Key Ownership

Every API key identifies an owner. Core uses this to enforce two security invariants:

1. **Haseef ownership** — When a haseef is created (`POST /api/haseefs`), Core records which API key created it (`apiKeyId`). Only that API key can read, update, or delete the haseef.

2. **Scope ownership** — When a scope registers tools (`PUT /api/scopes/:scope/tools`), Core records which API key registered it (`apiKeyId`). This lets downstream consumers (like Spaces) know who owns each scope.

**Scope attachment validation** — When `PATCH /api/haseefs/:id` updates the `scopes[]` array, Core checks:
- The API key must own the haseef
- Every scope being attached must either:
  - Belong to the same API key (same owner), OR
  - Be a platform scope (`apiKeyId = null` — e.g. `spaces`, `scheduler`)

This prevents User A from attaching User B's scope to their haseef, or modifying User B's haseef.

```typescript
// Pseudocode — scope attachment validation inside PATCH /api/haseefs/:id
async function validateScopeAttachment(haseefId: string, scopeNames: string[], apiKeyId: string) {
  const haseef = await db.haseef.findUnique({ where: { id: haseefId } });
  if (haseef.apiKeyId !== apiKeyId) throw new Error('Forbidden');

  for (const name of scopeNames) {
    const scope = await db.scope.findUnique({ where: { name } });
    if (!scope) throw new Error(`Scope "${name}" does not exist`);
    // Platform scopes (apiKeyId = null) are always allowed
    if (scope.apiKeyId !== null && scope.apiKeyId !== apiKeyId) {
      throw new Error(`Forbidden: scope "${name}" belongs to another owner`);
    }
  }
}
```

**`GET /api/scopes` response** includes `apiKeyId` per scope so Spaces can filter by user:

```json
[
  { "name": "spaces",       "connected": true, "apiKeyId": null,          "tools": [...] },
  { "name": "my-gmail",     "connected": true, "apiKeyId": "key-abc-123", "tools": [...] },
  { "name": "sara-twitter", "connected": true, "apiKeyId": "key-xyz-789", "tools": [...] }
]
```

Spaces server maps API key IDs to users and filters: each user only sees platform scopes + scopes registered by their own API keys.

---

## Scope Scenarios

### 1. Email Scope

```
Where it runs: Any server
Routes by: target.email
```

```typescript
const hsafa = new HsafaSDK({ coreUrl, apiKey, scope: 'email' });

// TOOLS
await hsafa.registerTools([
  { name: 'send_email', description: 'Send an email',
    input: { to: 'string', subject: 'string', body: 'string' } },
  { name: 'get_inbox', description: 'Get recent emails',
    input: { limit: 'number?' } },
  { name: 'reply_email', description: 'Reply to an email',
    input: { emailId: 'string', body: 'string' } },
  { name: 'search_email', description: 'Search emails by keyword',
    input: { query: 'string' } },
]);

// HANDLE
hsafa.onToolCall('send_email', async ({ to, subject, body }, ctx) => {
  await smtp.send({ from: ctx.haseef.profile.email, to, subject, body });
  return { sent: true };
});

hsafa.onToolCall('get_inbox', async ({ limit = 20 }, ctx) => {
  return await imap.fetch(ctx.haseef.profile.email, limit);
});

hsafa.onToolCall('reply_email', async ({ emailId, body }, ctx) => {
  const original = await imap.get(emailId);
  await smtp.send({
    from: ctx.haseef.profile.email,
    to: original.from,
    subject: `Re: ${original.subject}`,
    body,
    inReplyTo: emailId,
  });
  return { sent: true };
});

hsafa.onToolCall('search_email', async ({ query }, ctx) => {
  return await imap.search(ctx.haseef.profile.email, query);
});

// EVENTS
imap.on('mail', (email) => {
  hsafa.pushEvent({
    type: 'email_received',
    data: {
      from: email.from,
      subject: email.subject,
      body: email.text,
      emailId: email.id,
    },
    attachments: email.attachments?.map(a => ({
      type: 'file',
      mimeType: a.contentType,
      url: a.url,
      name: a.filename,
    })),
    target: { email: email.to },
  });
});

hsafa.connect();
```

---

### 2. WhatsApp Scope

```
Where it runs: Any server
Routes by: target.phone
```

```typescript
const hsafa = new HsafaSDK({ coreUrl, apiKey, scope: 'whatsapp' });

// TOOLS
await hsafa.registerTools([
  { name: 'send_message', description: 'Send a WhatsApp message',
    input: { to: 'string', text: 'string' } },
  { name: 'send_image', description: 'Send an image via WhatsApp',
    input: { to: 'string', imageUrl: 'string', caption: 'string?' } },
  { name: 'get_contacts', description: 'Get WhatsApp contacts list',
    input: {} },
]);

// HANDLE
hsafa.onToolCall('send_message', async ({ to, text }, ctx) => {
  await whatsappApi.sendMessage({
    from: ctx.haseef.profile.phone,
    to,
    text,
  });
  return { sent: true };
});

hsafa.onToolCall('send_image', async ({ to, imageUrl, caption }, ctx) => {
  await whatsappApi.sendImage({
    from: ctx.haseef.profile.phone,
    to,
    imageUrl,
    caption,
  });
  return { sent: true };
});

hsafa.onToolCall('get_contacts', async (_, ctx) => {
  return await whatsappApi.getContacts(ctx.haseef.profile.phone);
});

// EVENTS — WhatsApp webhook
app.post('/webhook/whatsapp', (req, res) => {
  const { from, to, text, media } = parseWebhook(req.body);

  hsafa.pushEvent({
    type: 'message',
    data: { from, text },
    attachments: media ? [{
      type: media.type === 'image' ? 'image' : 'file',
      mimeType: media.mimeType,
      url: media.url,
    }] : undefined,
    target: { phone: to },
  });

  res.sendStatus(200);
});

hsafa.connect();
```

---

### 3. Spaces Scope

```
Where it runs: Separate server (has frontend, SSE, DB)
Routes by: haseefId (space server manages membership)
```

```typescript
const hsafa = new HsafaSDK({ coreUrl, apiKey, scope: 'spaces' });

// TOOLS
await hsafa.registerTools([
  { name: 'send_space_message', description: 'Send a message in a space',
    input: { spaceId: 'string', text: 'string' } },
  { name: 'get_space_messages', description: 'Get recent messages from a space',
    input: { spaceId: 'string', limit: 'number?' } },
  { name: 'get_spaces', description: 'Get spaces you are a member of',
    input: {} },
  { name: 'send_confirmation', description: 'Ask a yes/no question in a space',
    input: { spaceId: 'string', question: 'string' } },
  { name: 'send_vote', description: 'Create a poll in a space',
    input: { spaceId: 'string', question: 'string', options: 'string[]' } },
]);

// HANDLE
hsafa.onToolCall('send_space_message', async ({ spaceId, text }, ctx) => {
  const msg = await db.message.create({
    data: { spaceId, text, entityId: ctx.haseef.id, type: 'agent' },
  });
  sse.emit(`space:${spaceId}`, 'message', msg);
  return { messageId: msg.id, sent: true };
});

hsafa.onToolCall('get_spaces', async (_, ctx) => {
  const memberships = await db.spaceMembership.findMany({
    where: { entityId: ctx.haseef.id },
    include: { space: true },
  });
  return memberships.map(m => ({ id: m.space.id, name: m.space.name }));
});

hsafa.onToolCall('get_space_messages', async ({ spaceId, limit = 30 }, ctx) => {
  const messages = await db.message.findMany({
    where: { spaceId },
    orderBy: { seq: 'desc' },
    take: limit,
    include: { entity: { select: { displayName: true, type: true } } },
  });
  return messages.reverse().map(m => ({
    from: m.entity.displayName,
    text: m.text,
    time: m.createdAt,
  }));
});

hsafa.onToolCall('send_confirmation', async ({ spaceId, question }, ctx) => {
  const msg = await db.message.create({
    data: {
      spaceId,
      text: question,
      entityId: ctx.haseef.id,
      type: 'confirmation',
      metadata: { status: 'pending' },
    },
  });
  sse.emit(`space:${spaceId}`, 'message', msg);
  return { messageId: msg.id, status: 'pending' };
});

// EVENTS — user sends a message in a space
app.post('/api/spaces/:id/messages', async (req, res) => {
  const message = await db.message.create({ data: req.body });
  res.json(message);

  // Find all haseefs in this space
  const agentMembers = await db.spaceMembership.findMany({
    where: { spaceId: req.params.id, entity: { type: 'agent' } },
    include: { entity: true },
  });

  for (const member of agentMembers) {
    hsafa.pushEvent({
      type: 'message',
      data: {
        spaceId: req.params.id,
        from: req.body.senderName,
        text: req.body.text,
      },
      haseefId: member.entity.haseefId,  // direct routing
    });
  }
});

// EVENTS — user responds to a confirmation
app.post('/api/messages/:id/respond', async (req, res) => {
  const msg = await db.message.update({
    where: { id: req.params.id },
    data: { metadata: { status: req.body.response } },
    include: { space: true },
  });
  res.json(msg);

  // Find the haseef who asked the question
  const asker = msg.entityId;
  hsafa.pushEvent({
    type: 'interaction',
    data: {
      spaceId: msg.spaceId,
      messageId: msg.id,
      response: req.body.response,
      respondedBy: req.body.respondedBy,
    },
    haseefId: asker,
  });
});

hsafa.connect();
```

---

### 4 + 5 + 6. Robot Body + Vision + Voice (Single Codebase)

```
Where it runs: ON THE ROBOT'S COMPUTER
Routes by: target.robotId
Three scopes, one process
```

```typescript
// robot-service/index.ts — runs on the robot's onboard computer
import { HsafaSDK } from '@hsafa/sdk';
import { Hardware } from './hardware';
import { Camera } from './camera';
import { TTS, STT } from './audio';

const ROBOT_ID = process.env.ROBOT_ID;  // "reachy-01"

const body = new HsafaSDK({ coreUrl, apiKey, scope: 'body' });
const vision = new HsafaSDK({ coreUrl, apiKey, scope: 'vision' });
const voice = new HsafaSDK({ coreUrl, apiKey, scope: 'voice' });

// ─── BODY SCOPE ─────────────────────────────────────────

await body.registerTools([
  { name: 'move', description: 'Move the robot',
    input: { direction: 'string', distance: 'number' } },
  { name: 'grab', description: 'Grab an object with hand',
    input: { hand: 'string' } },
  { name: 'release', description: 'Release held object',
    input: { hand: 'string' } },
  { name: 'wave', description: 'Wave hand as greeting',
    input: {} },
  { name: 'look_at', description: 'Turn head to look at direction',
    input: { direction: 'string' } },
  { name: 'get_position', description: 'Get current robot position and posture',
    input: {} },
]);

body.onToolCall('move', async ({ direction, distance }) => {
  await Hardware.move(direction, distance);
  return { moved: true, direction, distance };
});

body.onToolCall('grab', async ({ hand }) => {
  const success = await Hardware.grab(hand);
  return { grabbed: success };
});

body.onToolCall('wave', async () => {
  await Hardware.playAnimation('wave');
  return { waved: true };
});

body.onToolCall('look_at', async ({ direction }) => {
  await Hardware.lookAt(direction);
  return { looking: direction };
});

body.onToolCall('get_position', async () => {
  return await Hardware.getPosition();
});

// Body events
Hardware.onBump((data) => {
  body.pushEvent({
    type: 'bump_detected',
    data: { sensor: data.location, force: data.force },
    target: { robotId: ROBOT_ID },
  });
});

Hardware.onBatteryLow((level) => {
  body.pushEvent({
    type: 'battery_low',
    data: { level },
    target: { robotId: ROBOT_ID },
  });
});

// ─── VISION SCOPE ───────────────────────────────────────

await vision.registerTools([
  { name: 'capture_image', description: 'Take a photo from the camera',
    input: {} },
  { name: 'detect_objects', description: 'Detect objects in current view',
    input: {} },
  { name: 'find_person', description: 'Look for a specific person by name',
    input: { name: 'string' } },
  { name: 'read_text', description: 'Read text visible to the camera (OCR)',
    input: {} },
]);

vision.onToolCall('capture_image', async () => {
  const image = await Camera.capture();
  return { imageUrl: image.url, timestamp: Date.now() };
});

vision.onToolCall('detect_objects', async () => {
  const frame = await Camera.capture();
  const objects = await Camera.detectObjects(frame);
  return { objects };
  // [{ label: "cup", position: "left", distance: "0.5m", confidence: 0.94 }]
});

vision.onToolCall('find_person', async ({ name }) => {
  const result = await Camera.findPerson(name);
  return result
    ? { found: true, direction: result.direction, distance: result.distance }
    : { found: false };
});

vision.onToolCall('read_text', async () => {
  const frame = await Camera.capture();
  return await Camera.ocr(frame);
});

// Vision events — continuous watching
Camera.onPersonDetected((person) => {
  vision.pushEvent({
    type: 'person_detected',
    data: {
      name: person.recognized ? person.name : 'unknown',
      distance: person.distance,
      direction: person.direction,
    },
    target: { robotId: ROBOT_ID },
  });
});

Camera.onGesture((gesture) => {
  vision.pushEvent({
    type: 'gesture_detected',
    data: { gesture: gesture.type, from: gesture.personName },
    target: { robotId: ROBOT_ID },
  });
});

// ─── VOICE SCOPE ────────────────────────────────────────

await voice.registerTools([
  { name: 'speak', description: 'Say something out loud through the speaker',
    input: { text: 'string', emotion: 'string?' } },
  { name: 'listen', description: 'Listen for speech and return transcription',
    input: { duration: 'number?' } },
  { name: 'set_volume', description: 'Set speaker volume',
    input: { level: 'number' } },
]);

voice.onToolCall('speak', async ({ text, emotion }) => {
  await TTS.speak(text, { emotion: emotion || 'neutral' });
  return { spoken: true };
});

voice.onToolCall('listen', async ({ duration = 5 }) => {
  const transcript = await STT.listen(duration * 1000);
  return { text: transcript.text, confidence: transcript.confidence };
});

voice.onToolCall('set_volume', async ({ level }) => {
  await TTS.setVolume(level);
  return { volume: level };
});

// Voice events — someone speaks near the robot
STT.onSpeechDetected((speech) => {
  voice.pushEvent({
    type: 'speech_heard',
    data: {
      text: speech.transcript,
      speaker: speech.speakerId || 'unknown',
      confidence: speech.confidence,
    },
    target: { robotId: ROBOT_ID },
  });
});

// ─── CONNECT ALL THREE ──────────────────────────────────

await Promise.all([body.connect(), vision.connect(), voice.connect()]);
console.log(`Robot service connected for ${ROBOT_ID}`);
```

**Haseef profile for the robot:**
```json
{
  "name": "Atlas",
  "profile": { "robotId": "reachy-01" },
  "scopes": ["body", "vision", "voice"]
}
```

**How the LLM combines them naturally:**

```
Event: person_detected (vision) → name: "Husam", direction: "left"

LLM thinks: "Husam is here, I should greet him"
  → look_at("left")                    (body)
  → wave()                             (body)
  → speak("Hello Husam!")              (voice)

Event: speech_heard (voice) → text: "Can you grab that cup?"

LLM thinks: "He wants the cup, let me find it"
  → detect_objects()                    (vision) → cup on right, 0.5m
  → move("right", 0.5)                 (body)
  → grab("right_hand")                 (body)
  → speak("Got it!")                   (voice)
```

No integration code. The LLM is the integration layer.

---

### 7. Game Control Scope

```
Where it runs: Game server or alongside game client
Routes by: target.playerId
```

```typescript
const hsafa = new HsafaSDK({ coreUrl, apiKey, scope: 'game' });

await hsafa.registerTools([
  { name: 'move_character', description: 'Move character in direction',
    input: { direction: 'string', steps: 'number' } },
  { name: 'attack', description: 'Attack nearest enemy',
    input: { weapon: 'string' } },
  { name: 'use_item', description: 'Use an item from inventory',
    input: { item: 'string' } },
  { name: 'get_status', description: 'Get health, inventory, position',
    input: {} },
  { name: 'chat_in_game', description: 'Send a message in game chat',
    input: { text: 'string' } },
  { name: 'get_nearby', description: 'Get nearby players, enemies, items',
    input: { radius: 'number?' } },
]);

hsafa.onToolCall('move_character', async ({ direction, steps }, ctx) => {
  return await gameApi.move(ctx.haseef.profile.playerId, direction, steps);
});

hsafa.onToolCall('attack', async ({ weapon }, ctx) => {
  return await gameApi.attack(ctx.haseef.profile.playerId, weapon);
});

hsafa.onToolCall('get_status', async (_, ctx) => {
  return await gameApi.getStatus(ctx.haseef.profile.playerId);
});

hsafa.onToolCall('get_nearby', async ({ radius = 10 }, ctx) => {
  return await gameApi.getNearby(ctx.haseef.profile.playerId, radius);
});

// Game events
gameApi.on('damage_taken', (e) => {
  hsafa.pushEvent({
    type: 'damage_taken',
    data: { amount: e.damage, from: e.source, healthRemaining: e.hp },
    target: { playerId: e.playerId },
  });
});

gameApi.on('item_found', (e) => {
  hsafa.pushEvent({
    type: 'item_found',
    data: { item: e.item, rarity: e.rarity },
    target: { playerId: e.playerId },
  });
});

gameApi.on('player_nearby', (e) => {
  hsafa.pushEvent({
    type: 'player_nearby',
    data: { playerName: e.name, distance: e.distance, hostile: e.hostile },
    target: { playerId: e.targetPlayerId },
  });
});

gameApi.on('chat_message', (e) => {
  hsafa.pushEvent({
    type: 'chat_received',
    data: { from: e.playerName, text: e.text },
    target: { playerId: e.targetPlayerId },
  });
});

hsafa.connect();
```

---

### 8. Database Scope

```
Where it runs: Any server with DB access
Routes by: haseefId (from trigger payload or broadcast)
```

```typescript
const hsafa = new HsafaSDK({ coreUrl, apiKey, scope: 'database' });

await hsafa.registerTools([
  { name: 'query', description: 'Run a read-only SQL query',
    input: { sql: 'string' } },
  { name: 'get_record', description: 'Get a record by table and ID',
    input: { table: 'string', id: 'string' } },
  { name: 'create_record', description: 'Insert a new record',
    input: { table: 'string', data: 'object' } },
  { name: 'update_record', description: 'Update a record by ID',
    input: { table: 'string', id: 'string', data: 'object' } },
  { name: 'list_tables', description: 'List available database tables',
    input: {} },
]);

// Safety: whitelist allowed tables
const ALLOWED_TABLES = ['orders', 'customers', 'tickets', 'products', 'notes'];

hsafa.onToolCall('query', async ({ sql }) => {
  if (!sql.trim().toLowerCase().startsWith('select')) {
    return { error: 'Only SELECT queries allowed. Use create_record or update_record for writes.' };
  }
  const result = await pool.query(sql);
  return { rows: result.rows, count: result.rowCount };
});

hsafa.onToolCall('get_record', async ({ table, id }) => {
  if (!ALLOWED_TABLES.includes(table)) return { error: `Table "${table}" not accessible` };
  const result = await pool.query(`SELECT * FROM "${table}" WHERE id = $1`, [id]);
  return result.rows[0] || { error: 'Not found' };
});

hsafa.onToolCall('create_record', async ({ table, data }) => {
  if (!ALLOWED_TABLES.includes(table)) return { error: `Table "${table}" not accessible` };
  const keys = Object.keys(data);
  const values = Object.values(data);
  const placeholders = keys.map((_, i) => `$${i + 1}`);
  const result = await pool.query(
    `INSERT INTO "${table}" (${keys.join(', ')}) VALUES (${placeholders.join(', ')}) RETURNING *`,
    values
  );
  return { created: result.rows[0] };
});

hsafa.onToolCall('update_record', async ({ table, id, data }) => {
  if (!ALLOWED_TABLES.includes(table)) return { error: `Table "${table}" not accessible` };
  const keys = Object.keys(data);
  const values = Object.values(data);
  const setClause = keys.map((k, i) => `"${k}" = $${i + 1}`).join(', ');
  const result = await pool.query(
    `UPDATE "${table}" SET ${setClause} WHERE id = $${keys.length + 1} RETURNING *`,
    [...values, id]
  );
  return { updated: result.rows[0] };
});

hsafa.onToolCall('list_tables', async () => {
  return { tables: ALLOWED_TABLES };
});

// EVENTS — listen for DB changes via LISTEN/NOTIFY
const listener = await pool.connect();
await listener.query('LISTEN order_created');
await listener.query('LISTEN ticket_updated');
await listener.query('LISTEN customer_signup');

listener.on('notification', (msg) => {
  const payload = JSON.parse(msg.payload);

  // DB trigger must include haseefId or a profile field for routing
  hsafa.pushEvent({
    type: msg.channel,
    data: payload,
    haseefId: payload.assignedHaseefId,  // explicit haseef assignment in DB row
  });
});

hsafa.connect();
```

**Postgres trigger example (set up by the DB admin):**

```sql
CREATE OR REPLACE FUNCTION notify_order_created() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('order_created', json_build_object(
    'id', NEW.id,
    'customer', NEW.customer_name,
    'amount', NEW.total,
    'assignedHaseefId', NEW.assigned_haseef_id
  )::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER order_created_trigger
  AFTER INSERT ON orders
  FOR EACH ROW EXECUTE FUNCTION notify_order_created();
```

---

### 9. Company-Specific Logic

```
Where it runs: Company's own server
Routes by: haseefId (company knows the assignment)
```

```typescript
// Example: internal HR + CRM system
const hsafa = new HsafaSDK({ coreUrl, apiKey, scope: 'company' });

await hsafa.registerTools([
  { name: 'get_employee', description: 'Look up employee by name',
    input: { name: 'string' } },
  { name: 'submit_leave', description: 'Submit a leave request',
    input: { employeeId: 'string', from: 'string', to: 'string', reason: 'string' } },
  { name: 'check_leave_balance', description: 'Check remaining leave days',
    input: { employeeId: 'string' } },
  { name: 'get_customer', description: 'Look up customer by name or ID',
    input: { query: 'string' } },
  { name: 'create_ticket', description: 'Create a support ticket',
    input: { customerId: 'string', subject: 'string', description: 'string', priority: 'string' } },
  { name: 'get_org_chart', description: 'Get team structure',
    input: { department: 'string?' } },
]);

hsafa.onToolCall('get_employee', async ({ name }) => {
  return await companyApi.searchEmployee(name);
});

hsafa.onToolCall('submit_leave', async ({ employeeId, from, to, reason }) => {
  const result = await companyApi.submitLeave({ employeeId, from, to, reason });
  return { requestId: result.id, status: 'submitted' };
});

hsafa.onToolCall('get_customer', async ({ query }) => {
  return await companyApi.searchCustomer(query);
});

hsafa.onToolCall('create_ticket', async ({ customerId, subject, description, priority }) => {
  const ticket = await companyApi.createTicket({ customerId, subject, description, priority });
  return { ticketId: ticket.id, status: 'open' };
});

// Company events
companyApi.onWebhook('leave_approved', (event) => {
  hsafa.pushEvent({
    type: 'leave_approved',
    data: { employeeId: event.employeeId, from: event.from, to: event.to },
    haseefId: HR_HASEEF_ID,
  });
});

companyApi.onWebhook('ticket_escalated', (event) => {
  hsafa.pushEvent({
    type: 'ticket_escalated',
    data: { ticketId: event.ticketId, customer: event.customerName, reason: event.reason },
    haseefId: SUPPORT_HASEEF_ID,
  });
});

hsafa.connect();
```

---

## Event Format

Every event follows the same structure:

```typescript
interface PushEventPayload {
  // What happened
  type: string;                  // "message", "email_received", "bump_detected", etc.
  data: Record<string, any>;     // structured data about the event

  // Optional: binary attachments
  attachments?: Attachment[];

  // Routing (one of these is required)
  haseefId?: string;             // direct: "send to this haseef"
  target?: Record<string, string>; // resolve: "find haseef with this profile field"
}

interface Attachment {
  type: 'image' | 'audio' | 'file';
  mimeType: string;              // "image/jpeg", "audio/ogg", "application/pdf"
  url?: string;                  // URL to the data (preferred)
  base64?: string;               // inline base64 (small data only)
  name?: string;                 // filename
}
```

**Examples:**

```typescript
// Simple text event
{ type: 'message', data: { from: 'Husam', text: 'Hello' }, target: { phone: '+966...' } }

// Event with image attachment
{
  type: 'message',
  data: { from: 'Husam', text: 'Check this out' },
  attachments: [{ type: 'image', mimeType: 'image/jpeg', url: 'https://...' }],
  target: { phone: '+966...' }
}

// Event with voice note
{
  type: 'voice_message',
  data: { from: 'Husam', duration: 12 },
  attachments: [{ type: 'audio', mimeType: 'audio/ogg', url: 'https://...' }],
  target: { phone: '+966...' }
}

// Event with document
{
  type: 'email_received',
  data: { from: 'sara@company.com', subject: 'Q4 Report' },
  attachments: [{ type: 'file', mimeType: 'application/pdf', url: 'https://...', name: 'Q4-Report.pdf' }],
  target: { email: 'atlas@company.com' }
}

// Simple data event (no attachments)
{ type: 'sensor_update', data: { temperature: 24.5, humidity: 65 }, target: { robotId: 'reachy-01' } }

// Direct routing (no target lookup)
{ type: 'order_created', data: { orderId: '123', amount: 500 }, haseefId: 'support-haseef-id' }
```

---

## Tool Registration Format

```typescript
interface ToolDefinition {
  name: string;                  // tool name the LLM sees
  description: string;           // what the tool does (LLM reads this)
  input: Record<string, string>; // parameter name → type ("string", "number", "boolean", "object", "string[]", "number?")
}
```

Types support optional markers:
- `string` — required string
- `number?` — optional number
- `string[]` — array of strings
- `object` — JSON object

**Examples:**

```typescript
await hsafa.registerTools([
  {
    name: 'send_message',
    description: 'Send a WhatsApp message to a phone number',
    input: { to: 'string', text: 'string' },
  },
  {
    name: 'search_email',
    description: 'Search emails by keyword. Returns matching emails with subject, from, and date.',
    input: { query: 'string', limit: 'number?', folder: 'string?' },
  },
  {
    name: 'create_record',
    description: 'Insert a new record into a database table',
    input: { table: 'string', data: 'object' },
  },
]);
```

---

## Core API Routes

```
# Haseef CRUD (ownership-scoped: API key can only see/modify haseefs it created)
POST   /api/haseefs                              # Create haseef (records apiKeyId from x-api-key)
GET    /api/haseefs                              # List haseefs owned by this API key
GET    /api/haseefs/:id                          # Get haseef details (must own)
PATCH  /api/haseefs/:id                          # Update haseef (must own; scope attachment validated)
DELETE /api/haseefs/:id                          # Delete haseef (must own)

# Events (services push events to haseefs)
POST   /api/events                               # Push event (with haseefId or target)

# Scopes (services register tools)
PUT    /api/scopes/:scope/tools                  # Register/update tools (records apiKeyId as scope owner)
GET    /api/scopes                               # List all registered scopes (includes apiKeyId for filtering)
GET    /api/scopes/:scope/tools                  # List tools in a scope

# Actions (tool call dispatch)
GET    /api/scopes/:scope/actions/stream         # SSE: tool call requests for this scope
POST   /api/actions												      	  # Temporary: tool call requests via polling
POST   /api/actions/:actionId/result             # Submit tool call result

# Consciousness / Runs
GET    /api/haseefs/:id/stream                   # SSE: real-time thinking output
GET    /api/haseefs/:id/runs                     # Run history
GET    /api/haseefs/:id/runs/:runId              # Run details

# Memory
GET    /api/haseefs/:id/memory                   # Browse memories
POST   /api/haseefs/:id/memory                   # Add memory manually
PATCH  /api/haseefs/:id/memory/:key              # Edit memory
DELETE /api/haseefs/:id/memory/:key              # Delete memory
POST   /api/haseefs/:id/memory/search            # Semantic search memories

# Dashboard
GET    /api/dashboard/status                     # Overview: haseefs, scopes, connections

GET    /health
```

---

## Database Schema

### Core Tables

```prisma
model Haseef {
  id           String   @id @default(uuid())
  name         String   @unique
  description  String?
  apiKeyId     String                          // which API key created this haseef (ownership)
  profileJson  Json?    @db.JsonB              // { phone, email, robotId, ... }
  configJson   Json     @db.JsonB              // { model, instructions, ... }
  scopes       String[] @default([])           // ["whatsapp", "spaces", "body"]
  createdAt    DateTime @default(now())
  updatedAt    DateTime @updatedAt

  runs         Run[]
  episodic     EpisodicMemory[]
  semantic     SemanticMemory[]
  social       SocialMemory[]
  procedural   ProceduralMemory[]
}

model Scope {
  id          String   @id @default(uuid())
  name        String   @unique                // "whatsapp", "body", "email"
  apiKeyId    String?                         // which API key registered this scope (ownership, null = platform)
  connected   Boolean  @default(false)        // is the service currently connected?
  lastSeenAt  DateTime?                       // last heartbeat
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  tools       ScopeTool[]
}

model ScopeTool {
  id          String @id @default(uuid())
  scopeId     String
  name        String                          // "send_message"
  description String                          // "Send a WhatsApp message"
  inputSchema Json   @db.JsonB               // { to: "string", text: "string" }
  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt

  scope Scope @relation(fields: [scopeId], references: [id], onDelete: Cascade)
  @@unique([scopeId, name])
}

model Run {
  id               String    @id @default(uuid())
  haseefId         String
  status           String    @default("running")   // running, completed, interrupted, failed
  triggerScope     String?                          // "whatsapp", "spaces"
  triggerType      String?                          // "message", "email_received"
  summary          String?   @db.Text
  stepCount        Int       @default(0)
  promptTokens     Int       @default(0)
  completionTokens Int       @default(0)
  durationMs       Int       @default(0)
  startedAt        DateTime  @default(now())
  completedAt      DateTime?

  haseef Haseef @relation(fields: [haseefId], references: [id], onDelete: Cascade)
  @@index([haseefId])
}

model EpisodicMemory {
  id        String   @id @default(uuid())
  haseefId  String
  runId     String?
  summary   String   @db.Text
  context   Json?    @db.JsonB
  embedding Float[]                           // pgvector
  createdAt DateTime @default(now())

  haseef Haseef @relation(fields: [haseefId], references: [id], onDelete: Cascade)
  @@index([haseefId])
}

model SemanticMemory {
  id         String    @id @default(uuid())
  haseefId   String
  key        String
  value      String    @db.Text
  importance Int       @default(5)            // 1-10
  embedding  Float[]                          // pgvector
  recalledAt DateTime?
  createdAt  DateTime  @default(now())
  updatedAt  DateTime  @updatedAt

  haseef Haseef @relation(fields: [haseefId], references: [id], onDelete: Cascade)
  @@unique([haseefId, key])
  @@index([haseefId])
  @@index([haseefId, importance])
}

model SocialMemory {
  id              String   @id @default(uuid())
  haseefId        String
  entityName      String                      // "Husam", "Sara"
  observations    Json?    @db.JsonB          // preferences, style, traits
  relationship    String?  @db.Text
  lastInteraction DateTime?
  createdAt       DateTime @default(now())
  updatedAt       DateTime @updatedAt

  haseef Haseef @relation(fields: [haseefId], references: [id], onDelete: Cascade)
  @@unique([haseefId, entityName])
}

model ProceduralMemory {
  id         String   @id @default(uuid())
  haseefId   String
  trigger    String   @db.Text
  response   String   @db.Text
  confidence Float    @default(0.5)
  hitCount   Int      @default(0)
  createdAt  DateTime @default(now())
  updatedAt  DateTime @updatedAt

  haseef Haseef @relation(fields: [haseefId], references: [id], onDelete: Cascade)
  @@index([haseefId])
}
```

---

## Stack

| Component | Purpose |
|-----------|---------|
| **Node.js + TypeScript** | Core runtime |
| **Postgres + pgvector** | All storage + semantic search |
| **Redis** | Event wakeup (BRPOP), real-time streaming (Pub/Sub) |
| **Vercel AI SDK** (`ai`) | LLM integration, streamText, tool loop |
| **Prisma** | Database access |
| **Express** | HTTP API |
| **Zod** | Schema validation |

### Dependencies

```
ai, @ai-sdk/anthropic, @ai-sdk/openai, @ai-sdk/google, @ai-sdk/xai, @openrouter/ai-sdk-provider
@prisma/client, @prisma/adapter-pg, pg
ioredis
express, cors
zod
```

---

## Core File Structure

```
core/src/
  index.ts                          # Express server, route registration, startup
  middleware/
    auth.ts                         # x-api-key validation
  routes/
    haseefs.ts                      # Haseef CRUD, profile, scopes
    events.ts                       # POST /api/events — receive and route events
    scopes.ts                       # Scope + tool registration
    actions.ts                      # Action stream (SSE) + result submission
    memory.ts                       # Memory CRUD + search
    runs.ts                         # Run history
    dashboard.ts                    # Dashboard status endpoint
  lib/
    db.ts                           # Prisma client
    redis.ts                        # Redis connections
    coordinator.ts                  # Trigger haseefs, manage concurrency, interrupts
    invoker.ts                      # The think loop: perceive → think → act → remember
    event-router.ts                 # Resolve events to haseefs (by ID or profile target)
    tool-dispatcher.ts              # Route tool calls to services via SSE
    tool-builder.ts                 # ScopeTool rows → AI SDK tools
    prompt-builder.ts               # System prompt construction
    stream-publisher.ts             # Publish text deltas + tool events to Redis Pub/Sub
    model-registry.ts               # LLM provider registry
  memory/
    working.ts                      # Load working memory from source
    episodic.ts                     # Run summaries + pgvector search
    semantic.ts                     # Facts + pgvector search
    social.ts                       # Person models
    procedural.ts                   # Learned patterns
    reflection.ts                   # Post-run reflection + learning
    selection.ts                    # Per-run memory assembly (critical + relevant + fill)
  prebuilt-tools/
    done.ts                         # Signal run completion
    set-memories.ts                 # Store semantic memories
    delete-memories.ts              # Remove memories
    recall-memories.ts              # Search memories + episodic history
```

---

## How Everything Connects

### Service Connects

```
1. Service starts
2. SDK calls PUT /api/scopes/whatsapp/tools → registers tools in DB
3. SDK opens SSE to GET /api/scopes/whatsapp/actions/stream
4. Core marks scope "whatsapp" as connected
5. Service is ready — tools available, events can flow
```

### Event Arrives

```
1. WhatsApp webhook fires → service calls hsafa.pushEvent({
     type: 'message', data: { from, text }, target: { phone: '+966...' }
   })
2. SDK calls POST /api/events
3. Core event-router resolves target: finds haseef with profile.phone matching
4. Core checks: is "whatsapp" in this haseef's scopes list? Yes
5. Coordinator triggers the haseef
6. If haseef is already thinking → interrupt (abort current run)
7. Invoker runs:
   a. Load haseef profile + config
   b. Assemble memory (episodic + semantic + social search)
   c. Load tools from haseef's active scopes
   d. Build system prompt
   e. streamText() with LLM
8. LLM decides to call send_message({ to: '...', text: '...' })
9. Core tool-dispatcher sends action to WhatsApp service via SSE
10. Service handler runs, returns result
11. SDK posts result to POST /api/actions/{actionId}/result
12. Core passes result back to LLM
13. LLM calls done({ summary: '...' })
14. Post-run reflection saves episodic memory
```

### Tool Call Dispatch

```
Core (LLM calls tool)
  → tool-dispatcher identifies scope for this tool
  → sends action on SSE channel for that scope:
    {
      actionId: "abc-123",
      toolName: "send_message",
      args: { to: "+966559876543", text: "Hello!" },
      haseef: { id: "atlas-id", name: "Atlas", profile: { phone: "+966501234567", ... } }
    }
  → waits for result (with timeout)
  
Service (listening on SSE)
  → SDK receives action
  → routes to registered handler
  → handler executes: whatsappApi.send(...)
  → SDK posts result to POST /api/actions/{actionId}/result
  → Core receives result, passes to LLM
```

---

## Dashboard

The dashboard is a web UI for managing everything. Config lives in Postgres, so changes take effect immediately — no restarts.

### Features

| Feature | What it does |
|---------|-------------|
| **Haseef management** | Create, edit, delete haseefs. Edit profile, instructions, model, active scopes |
| **Scope overview** | See all connected services, their tools, connection status |
| **Memory browser** | Search, view, edit, delete a haseef's memories |
| **Run history** | See every run — trigger, tools called, summary, tokens, duration |
| **Live feed** | Real-time stream of what haseefs are doing right now |

### No Restarts

Everything is in the database:
- Change a haseef's profile → next run uses the new profile
- Toggle a scope on/off → next run includes/excludes those tools
- Edit instructions → next run uses the new instructions
- Change model → next run uses the new model

Haseefs are stateless function invocations. They load config fresh from DB on every trigger.

---

## Summary

| Aspect | How it works |
|--------|-------------|
| **Core** | Brain — thinks, remembers, decides. Owns no external credentials. |
| **Services** | Body — perceive and act. Own their own credentials and logic. |
| **SDK** | 3 methods: registerTools, onToolCall, pushEvent. Handles all HTTP/SSE. |
| **Routing** | By profile field (`target: { phone }`) or direct (`haseefId`). Core resolves. |
| **Scopes** | Named groups of tools. Services register globally. Haseefs activate per-scope. |
| **Config** | All in Postgres. Dashboard edits. Zero restarts. |
| **Memory** | Structured: episodic, semantic, social, procedural. pgvector search. |
| **Deployment** | Core = one server. Services = wherever they want. Connect via SDK. |
