/**
 * What has actually been run, and what it is allowed to claim.
 *
 * Separate from `Panels.tsx` on purpose. Those components describe
 * configuration and capability; these describe artifacts that exist on disk,
 * and this is the only place in the dashboard where a number may appear without
 * a "would be" attached to it.
 *
 * The `INFERENCE READY` state is why the component is shaped this way. An
 * analysis in that state produced real output and then failed validation — the
 * Nepal flood run wrote a probability raster, a mask and 17,739 polygons, then
 * failed the Track A/B distribution gate. It is shown with its verdict and
 * without its geometry. Presenting it as a result would be the exact failure
 * this project argues against; omitting it would erase a real finding.
 */

'use client';

import type { StudyArea } from '@/lib/study-areas';
import type { Analysis } from '@/lib/analyses';
import {
  EXPLANATIONS,
  STATUS_COLOUR,
  STATUS_MEANING,
  analysesFor,
  statusFor,
} from '@/lib/analyses';
import { SourceBadge } from '@/components/Panels';

function StatusPill({ status }: { status: Analysis['status'] }) {
  return (
    <span
      className="badge"
      style={{ background: STATUS_COLOUR[status], color: '#fff' }}
      title={STATUS_MEANING[status]}
    >
      {status}
    </span>
  );
}

function Provenance({ analysis }: { analysis: Analysis }) {
  const observation = analysis.observation ?? {};
  const rows: [string, string][] = [];

  if (observation.scene_id != null) {
    rows.push(['Scene', String(observation.scene_id)]);
    rows.push(['Acquired', String(observation.acquired_at ?? 'unknown')]);
    rows.push(['Platform', String(observation.platform ?? 'unknown')]);
    rows.push(['Provider', String(observation.provider ?? 'unknown')]);
    rows.push(['Radiometry', String(observation.radiometry ?? 'unknown')]);
    rows.push(['Resolution', `${String(observation.resolution_m ?? '?')} m`]);
  }
  if (observation.storm_name != null) {
    rows.push(['Storm', `${String(observation.storm_name)} ${String(observation.season ?? '')}`]);
    rows.push(['Best-track fixes', String(observation.n_fixes ?? '?')]);
    rows.push(['Dataset', String(observation.dataset ?? 'IBTrACS')]);
    rows.push(['First fix', String(observation.first_fix ?? 'unknown')]);
    rows.push(['Last fix', String(observation.last_fix ?? 'unknown')]);
  }
  if (analysis.model?.version != null) {
    rows.push(['Model', `${String(analysis.model.name)} ${String(analysis.model.version)}`]);
    if (analysis.model.bands != null) {
      rows.push(['Bands', (analysis.model.bands as string[]).join(', ')]);
    }
  }
  if (analysis.processing?.processed_at != null) {
    rows.push(['Processed', String(analysis.processing.processed_at)]);
  }
  if (analysis.exposure != null) {
    const exposure = analysis.exposure as Record<string, unknown>;
    rows.push(['Exposure', String(exposure.product ?? exposure.dataset ?? 'unknown')]);
  }
  if (analysis.vulnerability?.available === false) {
    // Said outright rather than left out: a risk figure computed without a
    // vulnerability term is an upper bound, and a reader needs to know that.
    rows.push(['Vulnerability', 'not available, excluded from R']);
  }
  if (analysis.overlays && analysis.overlays.length > 0) {
    const resolution = analysis.overlays[0]?.approx_resolution_m;
    rows.push([
      'Map layers',
      `${analysis.overlays.map((o) => o.label).join(', ')}${resolution ? ` (drawn at ~${Math.round(resolution)} m)` : ''}`,
    ]);
  }

  if (rows.length === 0) return null;
  return (
    <dl className="analysis-provenance">
      {rows.map(([label, value]) => (
        <div key={label} className="analysis-provenance-row">
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function AnalysesPanel({ area }: { area: StudyArea | undefined }) {
  const produced = area ? analysesFor(area.id) : [];
  const status = area ? statusFor(area.id) : 'CONFIGURED';

  return (
    <div className="panel">
      <div className="panel-title">
        Produced analyses <StatusPill status={status} />
      </div>

      {produced.length === 0 ? (
        <p className="muted">
          {area?.name ?? 'This area'} is configured and nothing has been produced for it. That is
          a statement about this repository, not about the place.
        </p>
      ) : (
        <ul className="analysis-list">
          {produced.map((analysis, index) => (
            <li key={`${analysis.hazard}-${analysis.kind}-${index}`} className="analysis-item">
              <div className="analysis-head">
                <SourceBadge kind={analysis.source_kind} />
                <span className="analysis-hazard">{analysis.hazard}</span>
                <StatusPill status={analysis.status} />
              </div>

              {analysis.headline && <p className="analysis-headline">{analysis.headline}</p>}

              {analysis.not_a_prediction && (
                <p className="analysis-warning">{analysis.not_a_prediction}</p>
              )}

              {analysis.displayable === false && analysis.withheld_reason && (
                <p className="analysis-warning">
                  <strong>Not drawn on the map. </strong>
                  {analysis.withheld_reason}
                </p>
              )}

              {analysis.validation?.verdict && (
                <div className="analysis-validation">
                  <span className="panel-eyebrow">
                    {analysis.validation.instrument ?? 'Validation'}
                  </span>{' '}
                  <span
                    className={
                      analysis.validation.may_proceed ? 'status-pill ok' : 'status-pill no-data'
                    }
                  >
                    {analysis.validation.verdict}
                  </span>
                  {analysis.validation.bands && (
                    <ul className="analysis-bands">
                      {analysis.validation.bands.map((band) => (
                        <li key={band.band}>
                          <code>{band.band}</code> — {band.verdict}
                          {band.mean_shift !== undefined &&
                            ` · mean shift ${band.mean_shift > 0 ? '+' : ''}${band.mean_shift.toFixed(2)} dB`}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              <Provenance analysis={analysis} />

              {analysis.detections && analysis.detections.length > 0 && (
                <p className="muted">
                  {analysis.detections.length} detection(s) · most recent{' '}
                  {analysis.detections[0]?.observed_at}
                </p>
              )}

              {analysis.artifacts_served === false && analysis.artifacts_not_served_because && (
                <p className="muted analysis-footnote">{analysis.artifacts_not_served_because}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * Per-band attribution for the flood model, with its own contradiction visible.
 *
 * The ratio band takes the largest attribution share and removing it costs
 * 0.0019 IoU (C2). A panel that showed the attribution alone would imply the
 * band matters, so the disagreement is shown beside it.
 */
export function ExplainabilityPanel() {
  const explanation = EXPLANATIONS[0];
  if (!explanation) {
    return (
      <div className="panel">
        <div className="panel-title">Model explanation</div>
        <p className="muted">
          No attribution report has been produced. Run <code>python -m ml.flood.explain</code>.
        </p>
      </div>
    );
  }

  const shares = Object.entries(explanation.attribution_share);
  const peak = Math.max(...shares.map(([, value]) => Math.abs(value)), 1e-9);

  return (
    <div className="panel">
      <div className="panel-title">
        Model explanation <SourceBadge kind={explanation.source_kind} />
      </div>
      <p className="muted">
        Integrated gradients and occlusion over {explanation.n_chips} held-out{' '}
        {explanation.region ?? 'test'} chips · {explanation.model}{' '}
        <code>{explanation.model_version}</code>
      </p>

      <ul className="attribution-list">
        {shares.map(([band, share]) => (
          <li key={band} className="attribution-row">
            <span className="attribution-band">
              <code>{band}</code>
            </span>
            <span className="attribution-bar">
              <span
                className="attribution-fill"
                style={{ width: `${(Math.abs(share) / peak) * 100}%` }}
              />
            </span>
            <span className="attribution-value">{(share * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>

      {explanation.methods_disagree_on.length > 0 && (
        <p className="analysis-warning">
          The two methods disagree on the sign of{' '}
          {explanation.methods_disagree_on.map((band) => (
            <code key={band}>{band}</code>
          ))}
          . That disagreement is information about the model, not an error in either method.
        </p>
      )}

      <p className="muted analysis-footnote">
        Attribution describes what the model used, not physical causation. This is a MODEL
        EXPLANATION, not an explanation of a flood.
      </p>
    </div>
  );
}
