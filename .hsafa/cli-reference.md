# Hsafa CLI Reference

> All CLI commands for managing skills (v7).

## Install

```bash
npm install -g @hsafa/cli
```

## Configuration

```bash
hsafa config set-server <url>      # e.g. https://spaces.hsafa.com or http://localhost:3005
hsafa config show
hsafa config reset
```

## Authentication (against the Spaces server)

```bash
hsafa auth login                              # Interactive (browser)
hsafa auth login --token <token>              # With existing token
hsafa auth login --email e --password p       # Non-interactive (CI)
hsafa auth whoami                             # Show current user
hsafa auth logout                             # Clear credentials
```

## Building a Custom Skill (with @hsafa/sdk)

```bash
# 1. Scaffold a project
hsafa skill init my-weather --lang typescript --starter blank

# 2. Configure environment
cd my-weather
# Edit .env: set HSAFA_CORE_KEY to your Core SECRET_KEY
npm install

# 3. Run it
hsafa skill dev   # delegates to: npm run dev
```

A scaffolded project uses these env vars:

| Var | Purpose |
|------|--------|
| `SKILL_NAME` | Skill name registered with Core |
| `HSAFA_CORE_URL` | Core URL (default `http://localhost:3001`) |
| `HSAFA_CORE_KEY` | Core's `SECRET_KEY` — the single shared API key |

## Managing Skill Instances (Spaces server)

These commands talk to the Spaces server, where users create configured **instances** of skill **templates**.

```bash
# Browse what templates exist
hsafa skill templates

# Create an instance from a template
hsafa skill create my_db --template database --display "My Postgres"

# List your instances
hsafa skill list

# Delete an instance
hsafa skill delete my_db -y

# Attach / detach an instance to/from a haseef
hsafa skill attach my_db --haseef atlas
hsafa skill detach my_db --haseef atlas

# Show all skills attached to a haseef
hsafa skill show --haseef atlas
```

Haseefs can be referenced by **name** (case-insensitive) or **UUID**.

## Removed Commands (v6 → v7)

| Old command | Why it's gone |
|-------------|---------------|
| `hsafa skill register --key hsk_scope_*` | v7 uses a single Core `SECRET_KEY`; there are no per-skill keys |
| `hsafa skill publish` | Marketplace publish API isn't part of v7 yet |
| `hsafa skill install <slug>` | No marketplace install in v7 yet |
