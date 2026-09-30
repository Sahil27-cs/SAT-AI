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
  kind:
    | 'model_inference'
    | 'historical_analysis'
    | 'observation'
    | 'model_explanation'
    | 'risk_analysis';
  status: AnalysisStatus;
  source_kind: SourceKind;
  headline?: string;
  /** False when the geometry is not drawn, for either reason below. */
  displayable?: boolean;
  /** Which reason: a failed validation, or a raster too large to ship. */
  withheld_because?: 'validation_failed' | 'too_large_to_serve' | null;
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
  /** Small lon/lat PNGs of raster results, each tagged with its layer control. */
  overlays?: MapOverlay[];
  vulnerability?: { available?: boolean; reason?: string } | null;
  detections?: FireDetection[];
  artifacts_on_disk?: Record<string, string>;
  artifacts_served?: boolean;
  artifacts_not_served_because?: string;
  caveats: string[];
}

/** Which layer control an overlay belongs to. */
export type OverlayLayer = 'flood' | 'wildfire' | 'cyclone' | 'risk' | 'exposure' | 'damage';

export interface MapOverlay {
  layer: OverlayLayer;
  label: string;
  url: string;
  coordinates: [number, number][];
  approx_resolution_m?: number;
  note?: string;
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

export interface ExperimentArm {
  arm: string;
  bands: number;
  iou: number;
  f1: number;
  precision: number;
  recall: number;
}

/**
 * One contribution, as the artifacts on disk describe it.
 *
 * Derived rather than declared. The dashboard previously carried a hand-written
 * table of C1-C4 that went stale three times over in a single working session:
 * it told readers C1 needed an API key after the key was set and the loop was
 * answering, that C2 had two arms after four had been trained, and that C3 had
 * not been executed after its precondition was measured.
 */
export interface RegisteredExperiment {
  id: string;
  title: string;
  hypothesis?: string;
  status: string;
  dataset?: string;
  split?: string;
  test_region?: string;
  arms_executed?: ExperimentArm[];
  arms_blocked?: Record<string, string>;
  spread_iou?: number;
  finding?: string;
  what_this_is?: string;
  what_this_is_not?: string;
  expected_direction?: string;
  blocker?: string | { missing?: string; what_would_unblock_it?: string; targets_checked?: unknown[] };
  headline?: unknown;
  loro_iou?: number;
  official_iou?: number;
  gap_iou?: number;
  instrument?: Record<string, unknown>;
  report?: string | null;
  caveats?: string[];
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
  experiments: RegisteredExperiment[];
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

export const EXPERIMENTS_REGISTER: RegisteredExperiment[] = catalogue.experiments ?? [];

/** An experiment counts as executed when it produced a result, not when it exists. */
export function executedExperiments(): RegisteredExperiment[] {
  return EXPERIMENTS_REGISTER.filter((e) => e.status.includes('EXECUTED'));
}

/**
 * Every overlay that may be drawn, from validated analyses only.
 *
 * The filter is the honesty boundary for rasters: an analysis that ran but did
 * not pass its validation has no overlay in this list, whatever is on disk.
 */
export function mapOverlays(): MapOverlay[] {
  return ANALYSED_AREAS.flatMap((a) => a.analyses)
    .filter((a) => a.displayable === true)
    .flatMap((a) => a.overlays ?? [])
    .filter((o) => o.coordinates?.length === 4);
}
