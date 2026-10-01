/** Static editorial content, kept out of the component tree so the page stays structural. */

export interface Scope { does: string[]; doesNot: string[] }

export const HAZARDS: Record<string, { title: string; lede: string; scope: Scope }> = {
  flood: {
    title: 'Flood',
    lede: 'Flood susceptibility from terrain and rainfall state, and flood extent detected from Sentinel-1 SAR. SAR is cloud-independent, which matters because floods happen under cloud.',
    scope: {
      does: [
        'Susceptibility mapping from terrain, drainage and rainfall climatology',
        'Extent detection from SAR backscatter change',
        'Affected-area and exposure statistics',
        'Rainfall-modulated operational risk index',
      ],
      doesNot: [
        'Forecast when flooding will occur',
        'Estimate flood depth',
        'Detect flooding beneath dense canopy — C-band cannot see through it',
        'Perform reliably in dense urban fabric — double-bounce inverts the signature',
      ],
    },
  },
  wildfire: {
    title: 'Wildfire',
    lede: 'Fire-danger estimation from fire-weather and fuel-state proxies, plus ingestion of NASA FIRMS active-fire detections — which are observations at satellite overpass time, not predictions.',
    scope: {
      does: [
        'Fire-danger index from temperature, humidity, wind and fuel moisture',
        'Ingestion of FIRMS VIIRS/MODIS detections as OBSERVATIONS',
        'Burned-area mapping via dNBR',
        'Post-fire vegetation recovery trajectories',
      ],
      doesNot: [
        'Predict where a fire will ignite',
        'Treat FIRMS detections as forecasts',
        'Separate agricultural residue burning from forest fire without land-cover stratification',
        'Detect fires under cloud or between overpasses',
      ],
    },
  },
  cyclone: {
    title: 'Cyclone / Extreme Weather',
    lede: 'Extreme wind and rainfall risk indices, historical track analysis from IBTrACS, and exposure assessment along an observed or supplied track. SAT-AI does not forecast cyclones.',
    scope: {
      does: [
        'Extreme-rainfall percentile anomaly against climatology',
        'Extreme-wind risk from ERA5 gust percentiles',
        'Historical track and landfall statistics from IBTrACS v4',
        'Population and built-up exposure within wind-radius buffers',
      ],
      doesNot: [
        'Forecast cyclone track',
        'Forecast intensity or landfall location',
        'Replace IMD cyclone warnings',
        'Derive cyclone position from optical imagery alone',
      ],
    },
  },
  damage: {
    title: 'Damage Assessment',
    lede: 'Post-event change detection from co-registered pre/post imagery, with ordinal damage classification. Change detected from imagery has not been verified on the ground.',
    scope: {
      does: [
        'Co-registration and radiometric normalisation of pre/post pairs',
        'Change-vector analysis and supervised change classification',
        'Ordinal building-damage classification',
        'Affected-area and land-cover transition statistics',
      ],
      doesNot: [
        'Estimate monetary damage — no asset-value data exists',
        'Estimate casualties',
        'Present satellite-detected change as ground-verified',
        'Transfer reliably to Indian building stock — the xBD domain gap is untested',
      ],
    },
  },
};

export const SCOPE_OVERALL: Scope = {
  does: [
    'Flood susceptibility mapping and SAR extent detection',
    'Fire-danger estimation; ingestion of active-fire observations',
    'Extreme wind and rainfall risk indices',
    'Post-event damage assessment from pre/post imagery',
    'Explanations tied to actual model attributions',
  ],
  doesNot: [
    'Predict earthquakes — in any form',
    'Forecast cyclone tracks or intensity',
    'Predict wildfire ignition',
    'Forecast flood timing or depth',
    'Issue or relay official warnings',
  ],
};

export const CONTRIBUTIONS: [string, string, string][] = [
  [
    'C1',
    'Measured grounding-violation rate and tool-invocation accuracy for an agent over EO model outputs',
    'PARTIAL / EVALUATED — 28-question fixed benchmark framework with refusal correctness and grounding validator. Deployed agent enforces 100% refusal on ungrounded/emergency queries.',
  ],
  [
    'C2',
    'Quantified degradation under induced modality loss',
    'PARTIALLY EXECUTED — 5 of 8 arms completed (sar_ratio 0.5230 vs sar 0.5211 IoU, delta -0.0019). Remaining arms (sar_rain, sar_dem, full) pending co-registered DEM and rainfall chips.',
  ],
  [
    'C3',
    'Quantified rural-to-urban SAR transfer gap in India',
    'PRECONDITION MEASURED / BLOCKED — Target distribution shift measured on Mumbai Sentinel-1 scene (+1.54 dB VH, +2.12 dB VV); performance gap blocked on lack of open labeled urban Indian flood dataset.',
  ],
  [
    'C4',
    'Reproducible sensitivity analysis for multi-hazard risk formulation',
    'VALIDATED & EXECUTED — One-at-a-time exponent variation across [0.5, 2.0] confirms spatial ranking stability (Spearman rho >= 0.98), while continuous thresholding drives up to 28% discrete band reassignment.',
  ],
];

export const ESTABLISHED: string[] = [
  'SAR + precipitation + DEM fusion is the dominant flood architecture (Kemarau et al. 2026, from 176 studies).',
  'Encoder–decoder segmentation of Sentinel-1 is standard (Jones et al. 2023; Akhyar et al. 2024).',
  'Deriving susceptibility labels from AI flood extents is published practice (Jones et al. 2023).',
  'LLM agents over remote-sensing tools are a recognised class — six named systems (Shang et al. 2026).',
];

export const LATENCY: [string, string, string][] = [
  ['FIRMS active fire', 'near-real-time', '~3 h'],
  ['GPM IMERG Early rainfall', 'near-real-time', '~4 h'],
  ['ERA5 weather', 'retrospective', '~5 days'],
  ['Sentinel-1 flood extent', 'revisit-limited batch', '6–12 days'],
  ['API response', 'precomputed lookup', '< 500 ms target'],
];

export const BOUNDARIES: string[] = [
  'No earthquake prediction. Not in any form. Earthquake scope is post-event damage assessment only.',
  'No cyclone track or intensity forecasting. That needs assimilated observations and NWP, which IMD operates.',
  'No wildfire ignition prediction. Fire danger is estimated; ignition is not predicted.',
  'No flood timing or depth forecasting.',
  'Not an official warning system, and not a substitute for one.',
  'Not real-time. Near-real-time for weather-driven indices; revisit-limited batch for satellite-derived extent.',
];

export const LIMITATIONS: string[] = [
  'SAR flood detection fails predictably in cities: double-bounce between building walls and standing water RAISES backscatter where the method expects a drop.',
  'C-band cannot see flooding beneath dense canopy.',
  'Smooth dry surfaces — tarmac, sand, dry riverbeds — mimic water’s low backscatter. Terrain priors suppress but do not eliminate this.',
  'Flood-susceptibility labels are generated by SAT-AI’s own extent model, so its errors propagate. Published practice, and a real weakness; both are stated.',
  'Vulnerability is a PROXY from building density, road access and elevation above drainage. It omits income, age structure, disability and tenure, and so under-represents the vulnerability of marginalised populations.',
  'Sen1Floods11 is small: 446 hand-labelled chips across 11 events, 68 of them Indian.',
  'No ground-survey validation exists anywhere in this project. Every evaluation is satellite against satellite.',
];

export const XAI_METHODS: [string, string, string][] = [
  ['XGBoost (susceptibility, fire danger)', 'TreeSHAP', 'Per-location feature attributions'],
  ['U-Net (flood, damage)', 'Per-modality occlusion + integrated gradients', '"29.8% VV, 16.8% VH, 53.4% ratio" (model attribution, not physical causation)'],
  ['Risk engine', 'Analytic decomposition', 'Exact H/E/V split — it is a closed-form product'],
  ['Fusion models', 'Modality-ablation attribution', 'Ties explanation directly to contribution C2'],
];

export const ALERT_PERMITTED: string[] = [
  '"Model-estimated flood risk is high based on available inputs."',
  '"SAR backscatter decreased by X dB over this area."',
  '"Observed rainfall is in the Nth percentile of climatology."',
];

export const ALERT_FORBIDDEN: string[] = [
  '"A flood will happen."',
  '"Evacuate immediately."',
  '"An official warning has been issued."',
  'Any emergency telephone number.',
];

/**
 * Questions the flood assistant is built to answer, and the ones it must
 * decline. The last three are refusal and scope cases on purpose: a demo that
 * only shows easy questions does not show the part that matters.
 */
export const SAMPLE_QUESTIONS: string[] = [
  'Explain the flood model',
  'What is the India test performance?',
  'What is the Mekong validation score?',
  'Why is the India IoU lower than Mekong?',
  'What dataset provides the ground truth?',
  'Explain VV, VH and the VV/VH ratio',
  'How does the U-Net predict flooding?',
  'What is the Otsu baseline?',
  'Why was this scene blocked?',
  'Explain the distribution gate',
  'What happened with the 74.8 km² result?',
  'Was 74.8 km2 actually flooded?',
  'Explain the latest flood analysis',
  'What satellite scene was used?',
  'What is the risk?',
  'Is this an official government warning?',
  'Can you predict whether Mumbai will flood tomorrow?',
];

export const DISCLAIMER =
  'SAT-AI risk levels are prototype research outputs from a student project. They are NOT official warnings and do not replace IMD, NDMA or State Disaster Management Authority advisories.';
