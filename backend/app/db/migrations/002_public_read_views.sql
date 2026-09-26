-- 002_public_read_views.sql — the read surface PostgREST is allowed to serve.
--
-- PostgREST serves only the schemas on its exposed list, and this project's
-- list is the default (`public`). Adding `satai` to it would publish
-- `scenes`, `model_versions` and -- worst -- `chat_turns` alongside the
-- catalogue, because exposure is per schema and not per table.
--
-- So nothing is exposed wholesale. Instead each table that is genuinely public
-- gets an explicit view in `public` under a `satai_` prefix. Adding a table to
-- the API surface then means writing a line here, which is a decision someone
-- makes on purpose, rather than a consequence of having created a table.
--
-- `security_invoker = true` is what keeps this from becoming a privilege
-- escalation. Without it a view runs as its owner, and row-level security on
-- the underlying table stops applying -- the view would hand out exactly the
-- rows RLS exists to withhold. With it, the caller's own permissions still
-- decide, and the view is a rename and nothing more.
--
-- The prefix also keeps SAT-AI out of the way of the other application sharing
-- this database instance.
--
-- `chat_turns` is the exception at the bottom of this file: it needs a view for
-- a different reason, and it is write-only.

CREATE OR REPLACE VIEW public.satai_regions
    WITH (security_invoker = true) AS
    SELECT * FROM satai.regions;

CREATE OR REPLACE VIEW public.satai_experiments
    WITH (security_invoker = true) AS
    SELECT * FROM satai.experiments;

CREATE OR REPLACE VIEW public.satai_hazard_results
    WITH (security_invoker = true) AS
    SELECT * FROM satai.hazard_results;

CREATE OR REPLACE VIEW public.satai_latest_hazard_results
    WITH (security_invoker = true) AS
    SELECT * FROM satai.latest_hazard_results;

CREATE OR REPLACE VIEW public.satai_explanations
    WITH (security_invoker = true) AS
    SELECT * FROM satai.explanations;

CREATE OR REPLACE VIEW public.satai_observations
    WITH (security_invoker = true) AS
    SELECT * FROM satai.observations;

CREATE OR REPLACE VIEW public.satai_model_versions
    WITH (security_invoker = true) AS
    SELECT * FROM satai.model_versions;

-- ---------------------------------------------------------------------------
-- The C1 audit log — write-only, and here only because PostgREST needs it.
--
-- PostgREST reaches only the schemas on its exposed list, and `satai` is
-- deliberately not on it. That applies to writes as much as reads, so the
-- serving plane cannot append a chat turn to `satai.chat_turns` directly no
-- matter which key it holds: the request 404s at the router before any
-- privilege is considered. This view is the one door, and it is a narrow one.
--
-- `security_invoker = true` again, so the caller's own rights decide. 003
-- grants INSERT on it to the service role and nothing else, and grants no
-- SELECT to anyone. A view you may insert into but not read is an unusual
-- shape and the right one here: the serving plane's only legitimate interest
-- is appending, and transcripts are read out of band when the C1 rate is
-- computed.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.satai_chat_turns
    WITH (security_invoker = true) AS
    SELECT * FROM satai.chat_turns;
