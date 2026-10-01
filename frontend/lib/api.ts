/**
 * Data layer.
 *
 * Reads directly from Supabase PostgREST for catalogue data (regions,
 * experiments) and from the FastAPI serving plane for hazard results and the
 * agent. Both are read-only from the browser; nothing here holds a secret.
 *
 * Every accessor can return an explicit "unavailable" state. That is not an
 * error path to be smoothed over — SAT-AI produces results on a batch schedule
 * tied to satellite revisit, so "no analysis has been computed for this region"
 * is a correct and informative answer, and the interface renders it as one
 * rather than showing a placeholder number.
 */

export const SUPABASE_URL =
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? 'https://akqhuzgekjsvrizysfmp.supabase.co';
export const SUPABASE_KEY =
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? 'sb_publishable_ZqxJZMUFFB1LQmcpV92b5w_66qDwpiZ';
/**
 * Serving-plane base URL. The deployed FastAPI backend is the default so the
 * dashboard works out of the box; NEXT_PUBLIC_API_URL overrides it for local
 * development or a self-hosted backend. This is a public endpoint, not a
 * credential.
 */
export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ??
  'https://sat-ai-api-chiragpednekar3-8808s-projects.vercel.app';

export type StudyRole = 'training' | 'transfer_evaluation' | 'candidate';

export interface Region {
  id: string;
  name: string;
  country: string;
  bbox: number[];
  area_km2: number;
  utm_epsg: string;
  tile_count: number | null;
  study_role: StudyRole;
  primary_hazards: string[];
  label_sources: string[];
  selection_rationale: string | null;
  caveats: string[];
}

export interface Experiment {
  id: string;
  name: string;
  contribution: string | null;
  dataset: string | null;
  split_protocol: string | null;
  model: string | null;
  metrics: Record<string, unknown>;
  status: 'pending' | 'running' | 'complete' | 'failed';
  notes: string[];
  completed_at: string | null;
}

export interface HazardResult {
  region_id: string;
  hazard: string;
  risk_index: number;
  risk_band: 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED';
  confidence: number | null;
  flooded_area_km2: number | null;
  population_exposed: number | null;
  provenance: {
    source_kind: string;
    source_id: string;
    version: string;
    scene_ids: string[];
    observed_at: string | null;
    computed_at: string | null;
    observation_age_hours: number | null;
    caveats: string[];
  };
}

/** A deliberate, explained absence — not a failure. */
export interface Unavailable {
  available: false;
  reason: string;
  what_would_produce_it: string;
  region_id?: string;
  hazard?: string;
}

export function isUnavailable(x: unknown): x is Unavailable {
  return typeof x === 'object' && x !== null && (x as Unavailable).available === false;
}

/**
 * PostgREST serves only the schemas on its exposed list, so the database
 * publishes an explicit read-only view surface in `public` under this prefix
 * rather than exposing the whole internal `satai` schema. Keeping the mapping
 * here means callers still ask for `regions`, not `satai_regions`.
 */
const TABLE_PREFIX = 'satai_';

async function pg<T>(table: string, query = ''): Promise<T[]> {
  const res = await fetch(`${SUPABASE_URL}/rest/v1/${TABLE_PREFIX}${table}?${query}`, {
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
    },
    cache: 'no-store',
  });
  if (!res.ok) throw new Error(`database ${res.status}: ${await res.text()}`);
  return res.json() as Promise<T[]>;
}

export const getRegions = () => pg<Region>('regions', 'select=*&order=id');
export const getExperiments = () => pg<Experiment>('experiments', 'select=*&order=id');

export async function getHazard(
  region: string,
  hazard: string
): Promise<HazardResult | Unavailable> {
  if (!API_URL) {
    return {
      available: false,
      reason:
        'The serving-plane API URL is not configured for this deployment, so hazard results cannot be fetched.',
      what_would_produce_it: 'Set NEXT_PUBLIC_API_URL to the deployed FastAPI backend.',
      region_id: region,
      hazard,
    };
  }
  try {
    const res = await fetch(`${API_URL}/api/v1/${hazard}?region=${region}`, {
      cache: 'no-store',
    });
    if (!res.ok) {
      return {
        available: false,
        reason: `The serving plane returned ${res.status} for ${hazard} over ${region}.`,
        what_would_produce_it: 'Check the backend /health endpoint.',
        region_id: region,
        hazard,
      };
    }
    return res.json();
  } catch (e) {
    return {
      available: false,
      reason: `Could not reach the serving plane: ${(e as Error).message}`,
      what_would_produce_it: 'Verify NEXT_PUBLIC_API_URL and that the backend is deployed.',
      region_id: region,
      hazard,
    };
  }
}

/** What the assistant asked the map to do, by calling show_on_map. */
export interface MapAction {
  region: string;
  bbox: number[];
  activate_layers: string[];
}

export interface ChatReply {
  answer: string;
  agent: string;
  route_confidence: number;
  route_method: string;
  tools_called: string[];
  grounded: boolean;
  degraded: boolean;
  provenance: { source_kind: string; source_id: string; caveats: string[] }[];
  /**
   * Map intents from this turn. The assistant expresses "show me the flooded
   * area" as a tool call rather than as prose, and the dashboard executes it —
   * which is what makes the agent part of the geospatial system rather than a
   * chat window beside it.
   */
  map_actions: MapAction[];
  notes: string[];
}

/** Longer than the backend's own budget (tool rounds plus one rewrite), so
    the server's answer -- including a degraded one -- arrives before this. */
const CHAT_TIMEOUT_MS = 90_000;

export async function askAgent(message: string, region: string | null): Promise<ChatReply> {
  if (!API_URL) throw new Error('NEXT_PUBLIC_API_URL is not configured.');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHAT_TIMEOUT_MS);
  let res: Response | null = null;
  const body = JSON.stringify({ message, region });

  // Preferred endpoint: /api/chat. Fallback: /api/v1/chat or same-origin rewrite.
  const endpoints = [
    `${API_URL}/api/chat`,
    `${API_URL}/api/v1/chat`,
    '/api/chat',
  ];

  let lastError: Error | null = null;

  for (const url of endpoints) {
    try {
      res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
        signal: controller.signal,
      });
      if (res.status === 404 && url !== endpoints[endpoints.length - 1]) {
        // Route not found at this path, try next endpoint
        continue;
      }
      break;
    } catch (e) {
      lastError = e as Error;
      if (lastError.name === 'AbortError') {
        clearTimeout(timer);
        throw new Error(
          'The assistant took too long to answer. Try again, or ask a shorter question.',
        );
      }
      // Continue to next endpoint (e.g. same-origin proxy if CORS failed)
    }
  }

  clearTimeout(timer);

  if (!res) {
    const msg = lastError?.message || '';
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError') || msg.includes('CORS')) {
      throw new Error(
        'SAT-AI backend is unreachable or CORS blocked. Please verify the API deployment.',
      );
    }
    throw new Error(`The SAT-AI backend could not be reached: ${msg || 'network error'}.`);
  }

  if (res.status === 429) {
    throw new Error('Too many questions in a short time. Wait a minute and ask again.');
  }
  if (res.status === 422) {
    throw new Error('That question could not be sent (validation error). Try rephrasing it.');
  }
  if (res.status === 500) {
    throw new Error('SAT-AI backend returned an internal error (500).');
  }
  if (res.status === 502 || res.status === 503 || res.status === 504) {
    throw new Error('SAT-AI backend service is temporarily unavailable. Please retry shortly.');
  }
  if (!res.ok) {
    throw new Error(`The SAT-AI backend returned an error (${res.status}).`);
  }

  let data: ChatReply;
  try {
    data = await res.json();
  } catch {
    throw new Error('SAT-AI backend returned a malformed response.');
  }

  return data;
}

/** Which model wrote this reply, read from the notes the backend attaches. */
export function answeredBy(reply: ChatReply): string | null {
  const note = reply.notes.find((n) => n.startsWith('model '));
  return note ? note.slice('model '.length) : null;
}

export async function getHealth(): Promise<Record<string, unknown> | null> {
  if (!API_URL) return null;
  try {
    const res = await fetch(`${API_URL}/health`, { cache: 'no-store' });
    if (res.ok) return res.json();
  } catch {
    // Cross-origin failed; try same-origin proxy
  }
  try {
    const fallback = await fetch('/health', { cache: 'no-store' });
    if (fallback.ok) return fallback.json();
  } catch {
    // Both failed
  }
  return null;
}

/** Data age in words. The distinction the UI exists to preserve. */
export function formatAge(hours: number | null): string {
  if (hours === null) return 'unknown';
  if (hours < 1) return 'under an hour';
  if (hours < 48) return `${Math.round(hours)} hours`;
  return `${Math.round(hours / 24)} days`;
}

export const BAND_COLOUR: Record<string, string> = {
  GREEN: '#2E7D32',
  YELLOW: '#F9A825',
  ORANGE: '#EF6C00',
  RED: '#C62828',
};

/**
 * How a value was produced. Rendered as a badge next to every number, because
 * the difference between a measurement, a model output and a configurable
 * index is the difference between what SAT-AI can and cannot claim.
 */
export const SOURCE_KIND_LABEL: Record<string, { label: string; tone: string; help: string }> = {
  observation: {
    label: 'OBSERVED',
    tone: 'obs',
    help: 'Measured by an instrument at satellite overpass time. Not a prediction.',
  },
  model: {
    label: 'MODEL OUTPUT',
    tone: 'model',
    help: 'Produced by a trained model from observed imagery. Not a direct measurement.',
  },
  derived: {
    label: 'DERIVED',
    tone: 'derived',
    help: 'Deterministic computation over other values, such as an index or ratio.',
  },
  index: {
    label: 'COMPOSITE INDEX',
    tone: 'index',
    help: 'A documented, configurable composite score. Not a trained prediction and not a forecast.',
  },
  reanalysis: {
    label: 'REANALYSIS',
    tone: 'derived',
    help: 'Model-assimilated historical fields such as ERA5. Retrospective, roughly five days behind.',
  },
  catalogue: {
    label: 'CATALOGUE',
    tone: 'derived',
    help: 'A curated record, such as a study-area definition or scene index.',
  },
};
