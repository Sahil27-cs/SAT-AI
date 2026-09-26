'use client';

/**
 * The panels that make the pipeline legible: analysis status, model status,
 * satellite provenance, the pipeline strip, and the honest empty state.
 *
 * Every one of these is capable of rendering "nothing here yet" as a first-class
 * state with a reason and a next step, because that is the state the system is
 * actually in for most regions. The design goal is that an empty dashboard
 * reads as *instrumented* rather than *broken* — a user should be able to see
 * which stage the pipeline stopped at and what would move it forward.
 */

import type { HazardResult, Unavailable } from '@/lib/api';
import { formatAge, SOURCE_KIND_LABEL } from '@/lib/api';
import type { HazardEvent, StudyArea } from '@/lib/study-areas';
import { READINESS_COPY, ROLE_LABEL, readiness, verifiedEvents } from '@/lib/study-areas';

/* ---------------------------------------------------------------- pipeline */

/**
 * Stage states are derived from what actually exists, not decorative.
 * `blocked` means a named prerequisite is missing and the panel says which —
 * this strip is the fastest way to see that the system is wired end to end and
 * stops at training, rather than being empty for no stated reason.
 */
export type StageState = 'ok' | 'partial' | 'blocked';

export interface Stage {
  name: string;
  state: StageState;
  detail: string;
}

export function pipelineFor(area: StudyArea | undefined, result: HazardResult | null): Stage[] {
  const events = area ? verifiedEvents(area) : [];
  return [
    {
      name: 'Satellite data',
      state: events.length > 0 ? 'partial' : 'blocked',
      detail:
        events.length > 0
          ? `${events.length} verified event with confirmed sensor coverage`
          : 'No verified acquisition configured for this region',
    },
    {
      name: 'Preprocessing',
      state: 'partial',
      detail: 'Band math, leakage-safe normalisation and region-disjoint splits implemented; not run over this AOI',
    },
    {
      name: 'Flood model',
      state: 'blocked',
      detail:
        'Classical baselines implemented (Otsu + HAND, per-pixel ensemble). The deep segmentation model is not written, and no model has been trained or scored',
    },
    {
      name: 'Risk engine',
      state: 'partial',
      detail: 'Implemented and sensitivity-analysed (C4); needs hazard, exposure and vulnerability rasters',
    },
    {
      name: 'Explainability',
      state: 'blocked',
      detail: 'Depends on a trained model',
    },
    {
      name: 'AI assistant',
      state: result ? 'ok' : 'partial',
      detail: 'Tools, router and grounding validator run; language layer needs ANTHROPIC_API_KEY',
    },
  ];
}

export function PipelineStrip({ stages }: { stages: Stage[] }) {
  return (
    <div className="pipeline" role="list" aria-label="Analysis pipeline status">
      {stages.map((stage, i) => (
        <div key={stage.name} className="pipeline-item" role="listitem">
          <div className={`pipeline-node ${stage.state}`}>
            <span className="pipeline-name">{stage.name}</span>
            <span className="pipeline-state">{stage.state}</span>
          </div>
          <p className="pipeline-detail">{stage.detail}</p>
          {i < stages.length - 1 && <span className="pipeline-arrow" aria-hidden="true" />}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------ analysis box */

export function AnalysisPanel({
  area,
  result,
}: {
  area: StudyArea | undefined;
  result: HazardResult | null;
}) {
  if (!area) return null;
  const state = readiness(area, result !== null);
  const copy = READINESS_COPY[state];
  const event = verifiedEvents(area)[0];

  return (
    <aside className="analysis-panel">
      <div className="panel-eyebrow">SAT-AI analysis</div>
      <h3 className="analysis-region">{area.name}</h3>
      <div className="analysis-country">
        {area.country}
        <span className={`role-chip ${area.studyRole}`} title={ROLE_LABEL[area.studyRole].help}>
          {ROLE_LABEL[area.studyRole].label}
        </span>
      </div>

      <div className={`readiness ${copy.tone}`}>
        <span className="readiness-dot" aria-hidden="true" />
        {copy.label}
      </div>
      <p className="analysis-detail">{copy.detail}</p>

      <dl className="analysis-facts">
        <div>
          <dt>Bounds</dt>
          <dd className="num">
            {area.bbox[0].toFixed(2)}, {area.bbox[1].toFixed(2)} &rarr; {area.bbox[2].toFixed(2)},{' '}
            {area.bbox[3].toFixed(2)}
          </dd>
        </div>
        <div>
          <dt>Area</dt>
          <dd className="num">{area.areaKm2.toLocaleString()} km&sup2;</dd>
        </div>
        <div>
          <dt>Analysis CRS</dt>
          <dd className="num">{area.utmEpsg}</dd>
        </div>
        <div>
          <dt>Hazards</dt>
          <dd>{area.primaryHazards.join(', ')}</dd>
        </div>
        {event && (
          <>
            <div>
              <dt>Latest observation</dt>
              <dd className="num">{event.occurredOn}</dd>
            </div>
            <div>
              <dt>Sensor</dt>
              <dd>{event.sensor}</dd>
            </div>
          </>
        )}
        {result && (
          <div>
            <dt>Observation age</dt>
            <dd className="num">{formatAge(result.provenance.observation_age_hours)}</dd>
          </div>
        )}
      </dl>

      {!result && (
        <p className="analysis-foot">
          No model output for this region. Risk indices and confidence appear here only when a
          batch run has produced them.
        </p>
      )}
    </aside>
  );
}

/* --------------------------------------------------------- model + sensor */

export function ModelStatus({
  name,
  version,
  dataset,
  split,
  testRegion,
  status,
  blocker,
  metrics,
}: {
  name: string;
  version: string;
  dataset: string;
  split: string;
  testRegion: string;
  status: 'READY' | 'NOT TRAINED' | 'BLOCKED';
  blocker?: string;
  metrics?: [string, string][];
}) {
  return (
    <div className="panel model-status">
      <div className="panel-head">
        <span className="panel-eyebrow">Model status</span>
        <span className={`status-pill ${status.toLowerCase().replace(' ', '-')}`}>{status}</span>
      </div>
      <div className="model-name">{name}</div>
      <dl className="kv">
        <div>
          <dt>Version</dt>
          <dd className="num">{version}</dd>
        </div>
        <div>
          <dt>Training dataset</dt>
          <dd>{dataset}</dd>
        </div>
        <div>
          <dt>Split protocol</dt>
          <dd>{split}</dd>
        </div>
        <div>
          <dt>Test region</dt>
          <dd>{testRegion}</dd>
        </div>
      </dl>
      {metrics && metrics.length > 0 ? (
        <table className="metrics">
          <tbody>
            {metrics.map(([k, v]) => (
              <tr key={k}>
                <td>{k}</td>
                <td className="num">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="panel-note">
          No metrics. A metric is reported here only after the model has been evaluated on a
          region-disjoint test set.
        </p>
      )}
      {blocker && (
        <p className="blocker">
          <strong>Blocked by:</strong> {blocker}
        </p>
      )}
    </div>
  );
}

export function SatellitePanel({ event }: { event: HazardEvent | undefined }) {
  if (!event) {
    return (
      <div className="panel">
        <span className="panel-eyebrow">Satellite observation</span>
        <p className="panel-note">
          No verified acquisition is configured for this region. An entry appears here only after
          the sensor, the acquisition date and a retrievable reference product have each been
          checked.
        </p>
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="panel-head">
        <span className="panel-eyebrow">Satellite observation</span>
        <span className="badge obs">OBSERVED</span>
      </div>
      <dl className="kv">
        <div>
          <dt>Sensor</dt>
          <dd>{event.sensor}</dd>
        </div>
        <div>
          <dt>Event date</dt>
          <dd className="num">{event.occurredOn}</dd>
        </div>
        {event.acquisitionUtc && (
          <div>
            <dt>Acquisition (UTC)</dt>
            <dd className="num">{event.acquisitionUtc.replace('T', ' ').replace('+00:00', '')}</dd>
          </div>
        )}
        <div>
          <dt>Status</dt>
          <dd>{event.displayStatus}</dd>
        </div>
      </dl>
      <p className="panel-note">{event.verificationNote}</p>
      {event.referenceProducts.length > 0 && (
        <div className="refs">
          <div className="refs-title">Reference products &mdash; other organisations&rsquo; delineations, not SAT-AI output</div>
          <ul className="tight">
            {event.referenceProducts.map((url) => (
              <li key={url}>
                <a href={url} target="_blank" rel="noopener noreferrer">
                  {new URL(url).hostname.replace('www.', '')}
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------ empty state */

/**
 * Replaces the full-width "NO HAZARD ANALYSES HAVE BEEN COMPUTED" slab.
 *
 * Same honesty, better use of the space: it names the state, gives the reason,
 * and offers the things a reader can actually do next instead of ending the
 * page on a dead end.
 */
export function DataStatus({
  data,
  area,
  onShowMethodology,
}: {
  data: Unavailable;
  area: StudyArea | undefined;
  onShowMethodology?: () => void;
}) {
  const event = area ? verifiedEvents(area)[0] : undefined;
  return (
    <div className="data-status">
      <div className="data-status-head">
        <span className="status-pill no-data">No current analysis</span>
        <span className="muted">{area?.name}</span>
      </div>
      <p className="data-status-reason">{data.reason}</p>
      <div className="data-status-what">
        <span className="panel-eyebrow">What would produce it</span>
        <p>{data.what_would_produce_it}</p>
      </div>
      {event && (
        <div className="data-status-event">
          <span className="panel-eyebrow">On file for this region</span>
          <p>
            {event.name} &middot; {event.occurredOn} &middot; {event.sensor}
          </p>
        </div>
      )}
      {onShowMethodology && (
        <button className="ghost" type="button" onClick={onShowMethodology}>
          View model methodology
        </button>
      )}
    </div>
  );
}

/* ----------------------------------------------------------------- shared */

export function SourceBadge({ kind }: { kind: string }) {
  const meta = SOURCE_KIND_LABEL[kind] ?? SOURCE_KIND_LABEL.derived;
  return (
    <span className={`badge ${meta.tone}`} title={meta.help}>
      {meta.label}
    </span>
  );
}
