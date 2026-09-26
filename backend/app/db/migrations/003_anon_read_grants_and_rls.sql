-- 003_anon_read_grants_and_rls.sql — who may read what.
--
-- The publishable ("anon") key ships in the browser bundle. It is designed for
-- untrusted clients, which is only true if the database treats it as one. This
-- file is where that treatment is written down.
--
-- Shape of the policy:
--
--   * SELECT only. No role reachable with the publishable key may INSERT,
--     UPDATE or DELETE anything, so a leaked key costs nothing beyond read
--     access to data that is already public.
--   * Row-level security is enabled on every table, including the ones that
--     are fully public. A table without RLS enabled is one `GRANT` away from
--     being readable in full; a table with RLS and an explicit permissive
--     policy states its intent and survives a careless grant.
--   * `chat_turns` has RLS enabled and NO policy. In Postgres that means no
--     row is visible to any role without BYPASSRLS. The absence of a policy is
--     the mechanism, not an oversight -- worth saying because "we forgot to
--     write one" and "we deliberately wrote none" look identical in a schema
--     dump. Only the service-role key, which is never in this repository,
--     writes or reads it.
--
-- Roles: `anon` and `authenticated` are Supabase's. On a plain Postgres they
-- may not exist, so each grant is guarded -- this file has to run in the local
-- Docker stack too, where the whole point is to reproduce production faithfully
-- rather than approximately.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        CREATE ROLE anon NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
        CREATE ROLE authenticated NOLOGIN;
    END IF;
END
$$;

-- Reach the schemas, but nothing in them by default.
GRANT USAGE ON SCHEMA satai  TO anon, authenticated;
GRANT USAGE ON SCHEMA public TO anon, authenticated;

-- ---------------------------------------------------------------------------
-- Public catalogue: readable, and read-only.
-- ---------------------------------------------------------------------------
GRANT SELECT ON
    satai.regions,
    satai.experiments,
    satai.hazard_results,
    satai.latest_hazard_results,
    satai.explanations,
    satai.observations,
    satai.model_versions
TO anon, authenticated;

GRANT SELECT ON
    public.satai_regions,
    public.satai_experiments,
    public.satai_hazard_results,
    public.satai_latest_hazard_results,
    public.satai_explanations,
    public.satai_observations,
    public.satai_model_versions
TO anon, authenticated;

ALTER TABLE satai.regions        ENABLE ROW LEVEL SECURITY;
ALTER TABLE satai.experiments    ENABLE ROW LEVEL SECURITY;
ALTER TABLE satai.hazard_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE satai.explanations   ENABLE ROW LEVEL SECURITY;
ALTER TABLE satai.observations   ENABLE ROW LEVEL SECURITY;
ALTER TABLE satai.model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE satai.scenes         ENABLE ROW LEVEL SECURITY;

DO $$
DECLARE
    target TEXT;
BEGIN
    FOREACH target IN ARRAY ARRAY[
        'regions', 'experiments', 'hazard_results',
        'explanations', 'observations', 'model_versions'
    ]
    LOOP
        EXECUTE format('DROP POLICY IF EXISTS %I ON satai.%I', target || '_public_read', target);
        EXECUTE format(
            'CREATE POLICY %I ON satai.%I FOR SELECT TO anon, authenticated USING (true)',
            target || '_public_read', target
        );
    END LOOP;
END
$$;

-- ---------------------------------------------------------------------------
-- Not public.
-- ---------------------------------------------------------------------------

-- `scenes` has RLS enabled above and gets no policy and no grant: the scene
-- identifiers a result depends on are served through the result's provenance,
-- so there is no reason to publish the whole catalogue.
REVOKE ALL ON satai.scenes FROM anon, authenticated;

-- The C1 audit log. RLS enabled, no policy for anon or authenticated -- see the
-- header. The publishable key can neither read it nor append to it.
ALTER TABLE satai.chat_turns ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON satai.chat_turns FROM anon, authenticated;
REVOKE ALL ON public.satai_chat_turns FROM anon, authenticated;

-- The serving plane appends one row per turn using the service-role key. On
-- Supabase `service_role` carries BYPASSRLS, so it needs the grant and no
-- policy; the guard on this table is that nothing else can reach it at all.
-- Guarded because a plain Postgres has no such role and these migrations have
-- to run in the local Docker stack too.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
        GRANT USAGE  ON SCHEMA satai         TO service_role;
        GRANT INSERT ON satai.chat_turns     TO service_role;
        GRANT USAGE  ON SEQUENCE satai.chat_turns_id_seq TO service_role;
        -- INSERT only, and no SELECT: appending is the serving plane's entire
        -- legitimate interest in this table. Transcripts are read out of band
        -- when the C1 violation rate is computed.
        GRANT INSERT ON public.satai_chat_turns TO service_role;
    END IF;
END
$$;

-- Future tables must be opted in explicitly rather than inheriting access.
ALTER DEFAULT PRIVILEGES IN SCHEMA satai REVOKE ALL ON TABLES FROM anon, authenticated;
