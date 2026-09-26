# Deployment

Live URLs, how the pieces fit, and how to redeploy.

---

## Live deployment

| Component | URL | Verified |
|---|---|---|
| Frontend | https://sat-ai-murex.vercel.app | Renders; 12 sections, MapLibre map |
| Backend | https://sat-ai-api-chiragpednekar3-8808s-projects.vercel.app | `/health` → `status: ok` |
| API docs | `…/docs` (OpenAPI) and `…/openapi.json` | Served by FastAPI |
| Database | Supabase PostgreSQL + PostGIS, project `akqhuzgekjsvrizysfmp` | 4 regions, 10 experiments |

Health check at the time of writing:

```json
{"status":"ok","version":"0.4.0","checks":{"api":"ok","database":"ok","llm":"not_configured"}}
```

`llm: not_configured` is accurate and deliberate: no `ANTHROPIC_API_KEY` is set
on the deployment, so `/api/v1/chat` returns structured tool output with
`degraded: true` rather than failing. Setting the key turns the language layer
on with no other change.

---

## Shape

```
  Browser
     │
     ├──────────────► Supabase PostgREST      catalogue reads (regions, experiments)
     │                                        publishable key, read-only under RLS
     │
     └──────────────► FastAPI on Vercel       hazard results, provenance, agent
                            │
                            └───────────────► Supabase PostgREST
```

The browser reads the catalogue directly because it is public, read-only data
and a second hop would add latency for nothing. Everything with provenance
semantics goes through the API, which is where the envelope contract is
enforced.

---

## Database access model

PostgREST serves only the schemas on its exposed list, and this project's list
is the default (`public`). Rather than expose the whole internal `satai`
schema — which would also publish `scenes`, `chat_turns` and `model_versions` —
the database publishes an explicit read-only view surface in `public`, prefixed
`satai_`:

```
public.satai_regions                → satai.regions
public.satai_experiments            → satai.experiments
public.satai_hazard_results         → satai.hazard_results
public.satai_latest_hazard_results  → satai.latest_hazard_results
public.satai_explanations           → satai.explanations
public.satai_observations           → satai.observations
public.satai_model_versions         → satai.model_versions
```

Each view is `security_invoker = true`, so the underlying tables' row-level
security still decides what a caller may read; the view adds no privilege of
its own. The prefix keeps SAT-AI out of the way of the other application
sharing this database instance.

`satai.chat_turns` has RLS enabled with **no policy and no grant**. Conversation
transcripts are not public data, and the absence of a policy is the mechanism
that keeps them private rather than an oversight.

Migrations live in [`backend/app/db/migrations/`](../backend/app/db/migrations/)
and are applied in filename order:

| File | What it does |
|---|---|
| `001_schema.sql` | Tables, indexes and the latest-result view, all in the `satai` schema |
| `002_public_read_views.sql` | The `satai_`-prefixed `security_invoker` views listed above |
| `003_anon_read_grants_and_rls.sql` | `SELECT`-only grants, RLS on every table, no policy on `chat_turns` |

These were previously named only as hosted migration identifiers, so the access
model this page describes could not be reproduced from a clone — and the one
schema file in the repository created unprefixed tables in the default schema,
which is not what production runs. There is now a single definition, and
`docker compose up -d db` applies it to a fresh local volume.

---

## Secrets

Nothing in the repository is a secret, and nothing secret is in the repository.

| Value | Where it lives | Why that is safe |
|---|---|---|
| Supabase **publishable** key | `.env.example`, code fallback, browser bundle | Designed for untrusted clients; read-only under RLS |
| Supabase **service-role** key | Nowhere in this repository | Never referenced by any code path |
| `ANTHROPIC_API_KEY` | Environment only, no fallback | Would be a real credential leak |
| CDSE / Earth Engine credentials | Environment only, no fallback | Same |

The publishable URL and key appear as *fallbacks* in `backend/api/index.py` and
`frontend/lib/api.ts` so a deployment without configured environment variables
degrades to read-only catalogue access rather than to a blank page. Environment
variables always win. No write credential has a fallback.

`.env` is git-ignored; `.env.example` documents every variable with a
placeholder.

---

## Redeploying

### Frontend

```bash
cd frontend
npm install
npm run build          # must pass before deploying
```

Deploy to the Vercel project `sat-ai` (framework `nextjs`). Files:
`package.json`, `tsconfig.json`, `next.config.mjs`, `app/layout.tsx`,
`app/globals.css`, `app/page.tsx`, `lib/api.ts`, `lib/content.ts`.

Optional environment variables, all with working defaults:
`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`,
`NEXT_PUBLIC_SUPABASE_ANON_KEY`.

### Backend

Deploy `backend/` to the Vercel project `sat-ai-api` (framework `fastapi`).
Files: `api/index.py`, `api/satai_agents.py`, `requirements.txt`, `vercel.json`.

Environment variables (all optional except the last):

| Variable | Default | Effect |
|---|---|---|
| `SUPABASE_URL` | public project URL | Catalogue endpoint |
| `SUPABASE_ANON_KEY` | publishable key | Read access |
| `SUPABASE_TABLE_PREFIX` | `satai_` | View-surface prefix |
| `CORS_ALLOW_ORIGINS` | `*` | Browser origins |
| `ANTHROPIC_API_KEY` | unset | **Turns the language layer on** |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Reasoning model |

Verify with:

```bash
curl https://sat-ai-api-chiragpednekar3-8808s-projects.vercel.app/health
```

`status: ok` requires `database: ok`. Anything else is reported as `degraded`
rather than hidden.

### Deployment protection

Both Vercel projects have SSO/deployment protection **disabled** so the URLs
are publicly reachable. A newly created project inherits the team default,
which is protection on — if a fresh deployment redirects to `vercel.com/login`,
that is why.

---

## Local development

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000

# Frontend
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

### Docker

```bash
docker compose up --build       # API on :8000, Postgres+PostGIS on :5432
```

The Dockerfile in `backend/` builds the serving plane only. The ML plane is not
containerised: it needs a GPU and a conda geospatial stack, and a container
that cannot run the model it contains is worse than no container.

---

## What is *not* deployed

- **No hazard results.** The database has zero rows in `hazard_results`, and
  the interface says so on every hazard panel with what would produce them. The
  inference pipeline needs the Sen1Floods11 download, Earth Engine credentials
  and a GPU; the environment that built this deployment had none of the three.
  Placeholder extents were not generated.
- **No scheduled jobs.** Batch inference is run manually until there are
  results worth refreshing.
- **No authentication.** Everything published is public read-only research
  output. There is nothing to protect and no user accounts to protect it for.

---

## Known deployment constraints

- **Environment variables could not be set through the available Vercel API
  token**, which lacks `projectEnvVars:create`. This is why the publishable
  Supabase values exist as code fallbacks rather than as environment variables.
  Setting them in the Vercel dashboard overrides the fallbacks with no code
  change, and is the right thing to do for any deployment that is not a
  student prototype.
- **The build environment's egress policy** blocks `*.vercel.app`,
  `*.supabase.co`, `catalogue.dataspace.copernicus.eu` and
  `storage.googleapis.com`. Deployment verification was therefore done through
  an external fetcher rather than from inside the build container, and the
  Sen1Floods11 download could not be performed there at all.
