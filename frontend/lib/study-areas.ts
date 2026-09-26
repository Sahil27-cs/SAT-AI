/**
 * Study areas, generated from `configs/aoi.yaml`.
 *
 * These are configuration, not results. They change when someone edits a YAML
 * file, not on a satellite revisit, so they are baked in at build time rather
 * than fetched — the hazard results the dashboard shows still come from the
 * serving plane. `tests/test_study_area_export.py` fails the build if the
 * generated file drifts from the YAML, so there is one source of truth even
 * though there are two copies of it.
 *
 * The distinction this module exists to protect: a region being *configured* is
 * not the same as a region having *analysis*. A study area with no verified
 * event and no computed hazard result is a research site, and the interface has
 * to say so rather than render an empty chart that reads as "no hazard here".
 */

import generated from './study-areas.generated.json';

export type StudyRole = 'training' | 'transfer_evaluation' | 'candidate';
export type Hazard = 'flood' | 'wildfire' | 'cyclone' | 'damage';

export interface HazardEvent {
  id: string;
  name: string;
  hazard: string;
  occurredOn: string;
  sensor: string;
  acquisitionUtc: string | null;
  referenceProducts: string[];
  /** True only when sensor, date and a retrievable product were each checked. */
  verified: boolean;
  verificationNote: string;
  /** What the interface is permitted to call this event. */
  displayStatus: string;
}

export interface StudyArea {
  id: string;
  name: string;
  country: string;
  /** [minLon, minLat, maxLon, maxLat] in EPSG:4326. */
  bbox: number[];
  centroid: number[];
  utmEpsg: string;
  areaKm2: number;
  tileCount: number;
  primaryHazards: string[];
  studyRole: StudyRole;
  status: string;
  labelSourcesDeclared: boolean;
  labelSources: string[];
  selectionRationale: string;
  notes: string | null;
  /** Per hazard: does a *verified* event back it? Drives the honest empty state. */
  hazardCoverage: Record<string, boolean>;
  events: HazardEvent[];
}

/**
 * Cast through `unknown` deliberately. TypeScript infers the imported JSON as a
 * union of each entry's exact literal shape — so `hazardCoverage` comes back as
 * `{ wildfire: boolean }` for the Uttarakhand entry and `{ flood: boolean }` for
 * Bihar, neither of which is assignable to `Record<string, boolean>`. The
 * declared interface above is the contract, and
 * `tests/test_study_area_export.py` is what actually enforces that the JSON
 * matches it — a structural check against the Python model, which is stricter
 * than anything the compiler can do against a generated file.
 */
export const STUDY_AREAS: StudyArea[] = (generated.studyAreas as unknown as StudyArea[]).slice();
export const CATALOGUE_VERSION: string = generated.version;

export function getStudyArea(id: string): StudyArea | undefined {
  return STUDY_AREAS.find((a) => a.id === id);
}

export function verifiedEvents(area: StudyArea): HazardEvent[] {
  return area.events.filter((e) => e.verified);
}

/** Areas that can offer historical analysis, as opposed to being configured. */
export function areasWithVerifiedEvents(): StudyArea[] {
  return STUDY_AREAS.filter((a) => verifiedEvents(a).length > 0);
}

export const ROLE_LABEL: Record<StudyRole, { label: string; help: string }> = {
  training: {
    label: 'TRAINING',
    help: 'The model is fitted on chips from this region. Metrics reported here are not generalisation.',
  },
  transfer_evaluation: {
    label: 'TRANSFER EVAL',
    help: 'Held out entirely. Measures how the model performs on a setting it never saw.',
  },
  candidate: {
    label: 'CANDIDATE',
    help: 'Configured but not yet promoted on measured evidence (ADR-007).',
  },
};

export const HAZARD_LABEL: Record<string, { label: string; tone: string }> = {
  flood: { label: 'Flood', tone: 'flood' },
  wildfire: { label: 'Wildfire', tone: 'fire' },
  cyclone: { label: 'Cyclone / extreme weather', tone: 'wind' },
  damage: { label: 'Post-event damage', tone: 'damage' },
};

/**
 * What a region can currently show, as a single honest verdict.
 *
 * Deliberately not a boolean. "Configured, no data" and "has a verified event
 * but no model output yet" are different states and collapsing them would make
 * the interface either over- or under-claim.
 */
export type Readiness = 'analysis' | 'event_only' | 'configured';

export function readiness(area: StudyArea, hasComputedResult: boolean): Readiness {
  if (hasComputedResult) return 'analysis';
  if (verifiedEvents(area).length > 0) return 'event_only';
  return 'configured';
}

export const READINESS_COPY: Record<Readiness, { label: string; tone: string; detail: string }> = {
  analysis: {
    label: 'ANALYSIS AVAILABLE',
    tone: 'ok',
    detail: 'A completed batch run has produced hazard output for this region.',
  },
  event_only: {
    label: 'HISTORICAL EVENT ON FILE',
    tone: 'warn',
    detail:
      'A dated event with confirmed satellite coverage is configured for this region, but the inference pipeline has not been run over it yet.',
  },
  configured: {
    label: 'CONFIGURED — NO ANALYSIS',
    tone: 'idle',
    detail:
      'This study area is defined and its data sources are identified. No verified event and no completed model run yet.',
  },
};
