/**
 * Analyses that have actually been produced, generated from artifacts on disk.
 *
 * `study-areas.ts` answers "what is configured". This answers "what has been
 * run", and the two are deliberately separate modules because conflating them
 * is how a dashboard comes to show seven study areas as though seven analyses
 * existed.
 *
 * The state that matters most here is `INFERENCE READY`. It means artifacts were
 * produced and did **not** pass validation — the Nepal flood run wrote a
 * probability raster, a mask and 17,739 extent polygons, and then failed the
 * Track A/B distribution gate because the model was trained on sigma0 and the
 * scene is gamma0. The record is shown; the extent is not. An unvalidated flood
 * map is the specific thing this project exists to argue against.
 */

import generated from './analyses.generated.json';

export type AnalysisStatus =
  | 'CONFIGURED'
  | 'DATA AVAILABLE'
  | 'MODEL READY'
  | 'INFERENCE READY'
  | 'ANALYSIS COMPLETE'
  | 'BLOCKED';

/** What kind of claim a layer is making. Never mixed without a label. */
export type SourceKind = 'observation' | 'model' | 'derived' | 'index' | 'catalogue';

export interface BandComparison {
  band: string;
  verdict: string;
  ks_statistic?: number;
  wasserstein?: number;
  mean_shift?: number;
  std_ratio?: number;
}

export interface Analysis {
  aoi: string;
  hazard: string;
  kind: 'model_inference' | 'historical_analysis' | 'observation' | 'model_explanation';
  status: AnalysisStatus;
  source_kind: SourceKind;
  headline?: string;
  /** False when the artifacts exist but did not pass validation. */
  displayable?: boolean;
  withheld_reason?: string | null;
  not_a_prediction?: string;
  observation?: Record<string, unknown>;
  model?: Record<string, unknown>;
  processing?: Record<string, unknown>;
  validation?: {
    instrument?: string;
    verdict?: string;
    may_proceed?: boolean;
    bands?: BandComparison[];
  };
  exposure?: Record<string, unknown>;
  wind?: Record<string, unknown>;
  risk?: Record<string, unknown>;
  population?: Record<string, unknown>;
  /** Servable geometry, present only where it is small enough to ship. */
  geometry?: { type: string; features: unknown[] } | null;
  detections?: FireDetection[];
  artifacts_on_disk?: Record<string, string>;
  artifacts_served?: boolean;
  artifacts_not_served_because?: string;
  caveats: string[];
}

export interface FireDetection {
  lat: number;
  lon: number;
  observed_at: string;
  frp_mw: number | null;
  brightness_k: number | null;
  confidence: string | null;
  satellite: string | null;
}

export interface AnalysedArea {
  id: string;
  name: string;
  country: string;
  bbox: number[];
  primary_hazards: string[];
  study_role: string;
  status: AnalysisStatus;
  analyses: Analysis[];
}

export interface Explanation {
  hazard: string;
  kind: string;
  status: AnalysisStatus;
  source_kind: SourceKind;
  model: string;
  model_version: string;
  region: string | null;
  n_chips: number;
  bands: string[];
  attribution_share: Record<string, number>;
  mean_integrated_gradients: Record<string, number>;
  mean_occlusion_delta: Record<string, number>;
  methods_disagree_on: string[];
  run_at: string;
  report: string;
  caveats: string[];
}

interface Catalogue {
  generated_at: string;
  status_vocabulary: Record<string, string>;
  study_areas: AnalysedArea[];
  wildfire: {
    window: string;
    window_days: number;
    run_at: string;
    provenance: Record<string, unknown>;
    totals: { areas_examined: number; areas_with_detections: number; detections: number };
    per_area: { aoi: string; detection_count: number; detections: FireDetection[] }[];
    caveats: string[];
  } | null;
  explanations: Explanation[];
}

const catalogue = generated as unknown as Catalogue;

export const ANALYSIS_CATALOGUE = catalogue;
export const ANALYSED_AREAS: AnalysedArea[] = catalogue.study_areas;
export const EXPLANATIONS: Explanation[] = catalogue.explanations ?? [];
export const WILDFIRE = catalogue.wildfire;

export function analysesFor(areaId: string): Analysis[] {
  return ANALYSED_AREAS.find((a) => a.id === areaId)?.analyses ?? [];
}

export function statusFor(areaId: string): AnalysisStatus {
  return ANALYSED_AREAS.find((a) => a.id === areaId)?.status ?? 'CONFIGURED';
}

/**
 * Analyses whose geometry may be drawn.
 *
 * The filter is the honesty boundary: everything else is described in text and
 * never rendered as a layer.
 */
export function displayableAnalyses(areaId?: string): Analysis[] {
  const pool = areaId
    ? analysesFor(areaId)
    : ANALYSED_AREAS.flatMap((a) => a.analyses);
  return pool.filter((a) => a.displayable === true);
}

/** Colour per state. Ordered by how much has actually been established. */
export const STATUS_COLOUR: Record<AnalysisStatus, string> = {
  CONFIGURED: '#6b7280',
  BLOCKED: '#b45309',
  'DATA AVAILABLE': '#0891b2',
  'MODEL READY': '#7c3aed',
  'INFERENCE READY': '#ca8a04',
  'ANALYSIS COMPLETE': '#15803d',
};

/** One line explaining what each state means, for a tooltip or a legend. */
export const STATUS_MEANING: Record<AnalysisStatus, string> = {
  CONFIGURED: 'Declared in configs/aoi.yaml. Nothing has been produced for it.',
  BLOCKED: 'A named dependency is missing. The blocker is listed with the area.',
  'DATA AVAILABLE': 'A real acquisition exists for this area.',
  'MODEL READY': 'A trained model applies to this hazard.',
  'INFERENCE READY':
    'Inference ran and wrote artifacts, but they did not pass validation, so nothing is drawn.',
  'ANALYSIS COMPLETE': 'Inference ran and passed its validation.',
};
