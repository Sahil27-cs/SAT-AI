'use client';

/**
 * SAT-AI dashboard. Two interface rules govern everything here:
 *
 * 1. A number is never shown without a source-kind badge saying how it was
 *    produced, and satellite-derived values carry their acquisition age.
 * 2. Absence is rendered honestly: where no analysis exists the panel says so
 *    and states what would produce it. No placeholders, no plausible figures.
 *    For a system whose contribution is about grounding, an invented number in
 *    the interface would undo the argument.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import 'maplibre-gl/dist/maplibre-gl.css';
import {
  API_URL, ChatReply, Experiment, HazardResult, Region, SOURCE_KIND_LABEL,
  Unavailable, answeredBy, askAgent, formatAge, getExperiments, getHazard, getHealth,
  getRegions, isUnavailable,
} from '@/lib/api';
import {
  ALERT_FORBIDDEN, ALERT_PERMITTED, BOUNDARIES, CONTRIBUTIONS, DISCLAIMER,
  ESTABLISHED, HAZARDS, LATENCY, LIMITATIONS, SAMPLE_QUESTIONS, SCOPE_OVERALL,
  Scope, XAI_METHODS,
} from '@/lib/content';
import {
  CATALOGUE_VERSION, ROLE_LABEL, STUDY_AREAS, StudyArea,
  areasWithVerifiedEvents, verifiedEvents,
} from '@/lib/study-areas';
import { CommandMap } from '@/components/CommandMap';
import {
  AnalysisPanel, DataStatus, ModelStatus, PipelineStrip, SatellitePanel, pipelineFor,
} from '@/components/Panels';
import { AnalysesPanel, ExplainabilityPanel } from '@/components/AnalysesPanel';
import type { RegisteredExperiment } from '@/lib/analyses';
import { ANALYSED_AREAS, EXPERIMENTS_REGISTER, executedExperiments } from '@/lib/analyses';

const SECTIONS = [
  { group: 'Monitoring', items: [['overview','Overview'],['map','Risk Map'],['flood','Flood'],
    ['wildfire','Wildfire'],['cyclone','Cyclone / Extreme Weather'],['damage','Damage Assessment']] },
  { group: 'Analysis', items: [['xai','Explainable AI'],['warning','Early Warning'],
    ['historical','Historical Analysis'],['assistant','Flood Assistant']] },
  { group: 'Research', items: [['research','Research & Methodology'],['about','About & Limitations']] },
] as const;

// --- primitives ---

const Badge = ({ kind }: { kind: string }) => {
  const m = SOURCE_KIND_LABEL[kind] ?? SOURCE_KIND_LABEL.derived;
  return <span className={`badge ${m.tone}`} title={m.help}>{m.label}</span>;
};

const Stat = ({ k, v, s }: { k: string; v: string; s?: string }) => (
  <div className="stat"><div className="k">{k}</div><div className="v">{v}</div>
    {s && <div className="s">{s}</div>}</div>
);

const NotAvailable = ({ data }: { data: Unavailable }) => (
  <div className="unavailable">
    <div className="h">No analysis available</div>
    <p className="r">{data.reason}</p>
    <div className="w">{data.what_would_produce_it}</div>
  </div>
);

const Disclaimer = () => (
  <div className="disclaimer"><strong>Not an official warning.</strong> {DISCLAIMER}</div>
);

const List = ({ items }: { items: string[] }) => (
  <ul className="tight">{items.map((x, i) => <li key={i}>{x}</li>)}</ul>
);

const ScopePanel = ({ scope, title }: { scope: Scope; title: string }) => (
  <div className="panel">
    <div className="panel-title">{title}</div>
    <div className="grid grid-2">
      <div><h3 style={{ marginTop: 0, color: 'var(--obs)' }}>Does</h3><List items={scope.does} /></div>
      <div><h3 style={{ marginTop: 0, color: 'var(--danger)' }}>Does not</h3><List items={scope.doesNot} /></div>
    </div>
  </div>
);

// --- hazard section ---

function HazardSection({ hazard, region }: { hazard: string; region: string }) {
  const [data, setData] = useState<HazardResult | Unavailable | null>(null);
  const meta = HAZARDS[hazard];

  // Guarded against out-of-order resolution. Switching regions quickly issues
  // overlapping requests, and without this the slower one can land last and
  // paint the previous region's numbers under the current region's heading —
  // a wrong number shown with full provenance, which is the worst failure this
  // interface has.
  useEffect(() => {
    let current = true;
    setData(null);
    getHazard(region, hazard).then((result) => { if (current) setData(result); });
    return () => { current = false; };
  }, [region, hazard]);

  return (
    <>
      <h2>{meta.title}</h2>
      <p className="lede">{meta.lede}</p>
      <Disclaimer />
      <ScopePanel scope={meta.scope} title="Scope — what this module does and does not do" />

      {data === null && <div className="panel muted">Loading…</div>}
      {data !== null && isUnavailable(data) && <NotAvailable data={data} />}
      {data !== null && !isUnavailable(data) && (
        <>
          <div className="pill-row">
            <Badge kind={data.provenance.source_kind} />
            <span className="muted">
              {data.provenance.source_id} v{data.provenance.version} · observation age{' '}
              {formatAge(data.provenance.observation_age_hours)}
            </span>
          </div>
          <div className="grid grid-4">
            <Stat k="Risk index" v={data.risk_index.toFixed(3)} s={`band ${data.risk_band}`} />
            <Stat k="Confidence" v={data.confidence !== null ? data.confidence.toFixed(2) : 'not reported'}
              s={data.confidence === null ? 'model provides none' : 'calibrated'} />
            <Stat k="Mapped extent" v={data.flooded_area_km2 !== null ? `${data.flooded_area_km2.toFixed(1)} km²` : '—'} />
            <Stat k="Population exposed" v={data.population_exposed !== null ? data.population_exposed.toLocaleString() : '—'} />
          </div>
          <div className="panel" style={{ marginTop: 14 }}>
            <div className="panel-title">Provenance</div>
            <table><tbody>
              <tr><td>Scenes</td><td className="num">{data.provenance.scene_ids.join(', ') || '—'}</td></tr>
              <tr><td>Observed at</td><td className="num">{data.provenance.observed_at ?? '—'}</td></tr>
              <tr><td>Computed at</td><td className="num">{data.provenance.computed_at ?? '—'}</td></tr>
            </tbody></table>
            <h3>Caveats</h3><List items={data.provenance.caveats} />
          </div>
        </>
      )}
    </>
  );
}

// --- region selector ---

/**
 * Cards rather than a dropdown. With seven regions across two countries, three
 * study roles and four hazards, the selector is carrying real information —
 * which regions have a verified event, which are held-out transfer targets —
 * and a `<select>` throws all of it away.
 */
function RegionSelector({
  areas, selected, onSelect,
}: { areas: StudyArea[]; selected: string; onSelect: (id: string) => void }) {
  return (
    <section style={{ marginBottom: 20 }}>
      <div className="pill-row" style={{ marginBottom: 10 }}>
        <span className="panel-eyebrow">Study area</span>
        <span className="muted">
          {areas.length} configured · {new Set(areas.map((a) => a.country)).size} countries
        </span>
      </div>
      <div className="region-grid">
        {areas.map((a) => {
          const events = verifiedEvents(a);
          return (
            <button
              key={a.id}
              type="button"
              className={`region-card ${a.id === selected ? 'active' : ''}`}
              onClick={() => onSelect(a.id)}
              aria-pressed={a.id === selected}
            >
              <div className="region-card-top">
                <span className="region-card-name">{a.name}</span>
                <span className={`role-chip ${a.studyRole}`} title={ROLE_LABEL[a.studyRole].help}>
                  {ROLE_LABEL[a.studyRole].label}
                </span>
              </div>
              <span className="region-card-country">{a.country}</span>
              <div className="region-card-meta">
                {a.primaryHazards.map((h) => (
                  <span key={h} className="region-card-haz">{h}</span>
                ))}
              </div>
              {events.length > 0 ? (
                <span className="region-card-event">
                  ✓ {events[0].occurredOn} · {events[0].sensor.split(' ')[0]}
                </span>
              ) : (
                <span className="muted" style={{ fontSize: 10.5 }}>no verified event</span>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}

// --- map ---

function RiskMap({ regions }: { regions: Region[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const container = ref.current;
    if (!container || regions.length === 0) return;

    let map: import('maplibre-gl').Map | undefined;
    // The cleanup below runs synchronously, but `map` is only assigned after
    // the dynamic import resolves. Unmounting during that window (React
    // StrictMode does it on every mount in development) would leave a live map
    // attached to a detached node, holding its WebGL context and tile requests.
    // This flag lets the async path see that it has been cancelled.
    let cancelled = false;

    (async () => {
      try {
        const maplibre = (await import('maplibre-gl')).default;
        if (cancelled) return;
        map = new maplibre.Map({
          container,
          style: {
            version: 8,
            sources: { osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
              tileSize: 256, attribution: '© OpenStreetMap contributors' } },
            layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
          },
          center: [82, 22], zoom: 3.6,
        });
        map.addControl(new maplibre.NavigationControl(), 'top-right');

        map.on('load', () => {
          const features = regions.map((r) => ({
            type: 'Feature' as const,
            properties: { name: r.name, role: r.study_role, area: r.area_km2,
              colour: r.study_role === 'transfer_evaluation' ? '#a78bfa' : '#38bdf8' },
            geometry: { type: 'Polygon' as const, coordinates: [[
              [r.bbox[0], r.bbox[1]], [r.bbox[2], r.bbox[1]],
              [r.bbox[2], r.bbox[3]], [r.bbox[0], r.bbox[3]], [r.bbox[0], r.bbox[1]]]] },
          }));
          map!.addSource('aoi', { type: 'geojson', data: { type: 'FeatureCollection', features } });
          map!.addLayer({ id: 'aoi-fill', type: 'fill', source: 'aoi',
            paint: { 'fill-color': ['get', 'colour'], 'fill-opacity': 0.18 } });
          map!.addLayer({ id: 'aoi-line', type: 'line', source: 'aoi',
            paint: { 'line-color': ['get', 'colour'], 'line-width': 2 } });

          map!.on('click', 'aoi-fill', (e) => {
            const f = e.features?.[0];
            if (!f) return;
            // Built as DOM rather than an HTML string. The region name comes
            // from the database, and interpolating it into setHTML makes any
            // future catalogue row an injection vector into every visitor's
            // browser. setDOMContent with textContent cannot be escaped out of,
            // so the safety does not depend on who edits the catalogue later.
            const popup = document.createElement('div');
            popup.className = 'map-popup';

            const name = document.createElement('strong');
            name.textContent = String(f.properties!.name ?? 'Unnamed area');

            const role = document.createElement('div');
            role.textContent = `role: ${String(f.properties!.role ?? 'unspecified')}`;

            const area = document.createElement('div');
            const km2 = Number(f.properties!.area);
            area.textContent = Number.isFinite(km2)
              ? `area: ${km2.toLocaleString()} km²`
              : 'area: not recorded';

            const note = document.createElement('em');
            note.textContent = 'No hazard analysis computed yet';

            popup.append(name, role, area, note);
            new maplibre.Popup({ closeButton: true })
              .setLngLat(e.lngLat)
              .setDOMContent(popup)
              .addTo(map!);
          });
          map!.on('mouseenter', 'aoi-fill', () => { map!.getCanvas().style.cursor = 'pointer'; });
          map!.on('mouseleave', 'aoi-fill', () => { map!.getCanvas().style.cursor = ''; });
        });
      } catch (e) { if (!cancelled) setError((e as Error).message); }
    })();

    return () => { cancelled = true; map?.remove(); };
  }, [regions]);

  if (error) return <div className="unavailable"><div className="h">Map unavailable</div><p className="r">{error}</p></div>;

  return (
    <div className="map-wrap">
      <div ref={ref} style={{ width: '100%', height: '100%' }} />
      <div className="map-legend">
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Study areas</div>
        <div className="row"><span className="swatch" style={{ background: '#38bdf8' }} />Training candidate</div>
        <div className="row"><span className="swatch" style={{ background: '#a78bfa' }} />Transfer evaluation</div>
        <div style={{ marginTop: 8, color: 'var(--fg-faint)', maxWidth: 220, lineHeight: 1.4 }}>
          Hazard layers appear once the inference pipeline has run. No placeholder extents are drawn.
        </div>
      </div>
    </div>
  );
}

// --- assistant ---

function Assistant({
  regions, region, onMapAction,
}: {
  regions: Region[];
  region: string;
  /** Fired when the agent calls show_on_map. This is what puts the assistant
      inside the geospatial system rather than beside it. */
  onMapAction?: (regionId: string, layers: string[]) => void;
}) {
  const [log, setLog] = useState<
    { role: 'user' | 'agent' | 'error'; text: string; meta?: ChatReply }[]
  >([]);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);

  const send = useCallback(async (text: string) => {
    if (!text.trim() || busy) return;
    setLog((l) => [...l, { role: 'user', text }]);
    setInput(''); setBusy(true);
    try {
      const reply = await askAgent(text, region);
      setLog((l) => [...l, { role: 'agent', text: reply.answer, meta: reply }]);
      // Execute the last map intent of the turn. The agent may call show_on_map
      // more than once while reasoning; the final one is what it settled on.
      const action = reply.map_actions?.[reply.map_actions.length - 1];
      if (action && onMapAction) onMapAction(action.region, action.activate_layers);
    } catch (e) {
      setLog((l) => [...l, { role: 'error', text: (e as Error).message }]);
    } finally { setBusy(false); }
  }, [busy, region, onMapAction]);

  return (
    <>
      <h2>SAT-AI Flood Assistant</h2>
      <p className="lede">
        Ask about the flood model, its data, its scores and the scenes it has been run on. Every
        project-specific value in a reply comes from a tool call against the serving plane, and a
        validator rejects any reply whose numbers are not traceable to a tool result. Flood only:
        wildfire, cyclone and earthquake questions are out of scope here.
      </p>
      <Disclaimer />

      {!API_URL ? (
        <div className="unavailable">
          <div className="h">Agent unavailable</div>
          <p className="r">The serving-plane API URL is not configured for this deployment.</p>
          <div className="w">Set NEXT_PUBLIC_API_URL to the deployed FastAPI backend, then redeploy.</div>
        </div>
      ) : (
        <div className="panel">
          <div className="panel-title">Try: including questions it must refuse</div>
          <div>{SAMPLE_QUESTIONS.map((q) => (
            <button key={q} className="chip" onClick={() => send(q)}>{q}</button>))}</div>

          {/* aria-live so a screen reader announces the reply when it lands.
              Without it the answer arrives silently and the user has to go
              looking for it. */}
          <div className="chat-log" style={{ marginTop: 14 }} aria-live="polite" aria-atomic="false">
            {log.length === 0 && (
              <div className="muted" style={{ padding: '30px 0', textAlign: 'center' }}>
                Ask about the flood model, or try one of the refusal cases above.
              </div>
            )}
            {log.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div className="who">
                  {m.role === 'user' ? 'You' : m.role === 'error' ? 'Not answered' : 'SAT-AI'}
                </div>
                <div className="body">{m.text}</div>
                {m.meta && (
                  <div className="meta">
                    <span>agent: {m.meta.agent}</span>
                    <span>route: {m.meta.route_method} ({m.meta.route_confidence})</span>
                    <span>tools: {m.meta.tools_called.join(' → ') || 'none'}</span>
                    {answeredBy(m.meta) && <span>model: {answeredBy(m.meta)}</span>}
                    {m.meta.map_actions?.length > 0 && (
                      <span style={{ color: 'var(--accent)' }}>
                        map → {m.meta.map_actions[m.meta.map_actions.length - 1].region}
                      </span>
                    )}
                    <span style={{ color: m.meta.grounded ? 'var(--obs)' : 'var(--danger)' }}>
                      {m.meta.grounded ? 'grounded' : 'GROUNDING VIOLATION'}</span>
                    {m.meta.degraded && <span style={{ color: 'var(--index)' }}>degraded: tool output only</span>}
                  </div>
                )}
              </div>
            ))}
            {busy && (
              <div className="muted" role="status">
                Checking SAT-AI&apos;s flood tools… this can take up to a minute.
              </div>
            )}
          </div>

          {/* A real form, so Enter submits natively and assistive technology
              announces it as one. The placeholder is not a label: it disappears
              on focus, which is exactly when a screen-reader user needs it. */}
          <form
            className="chat-input"
            onSubmit={(e) => { e.preventDefault(); send(input); }}
          >
            <label className="sr-only" htmlFor="assistant-input">
              Ask the SAT-AI Flood Assistant a question
            </label>
            <input id="assistant-input" name="question" value={input} disabled={busy}
              autoComplete="off"
              placeholder="Ask about the flood model, its metrics or the Nepal scene…"
              onChange={(e) => setInput(e.target.value)} />
            <button className="primary" type="submit" disabled={busy || !input.trim()}>Ask</button>
          </form>
        </div>
      )}
    </>
  );
}

// --- research ---

/**
 * C1–C4 with the state each is actually in and what gates it.
 *
 * `result` is the load-bearing field: for anything not executed it names the
 * blocker rather than leaving a blank, because a blank reads as "coming soon"
 * and a named blocker reads as a research register. Only C4 carries numbers,
 * because C4 is the only one that has run.
 */
/**
 * Presentation for a derived experiment state.
 *
 * The states come from `analyses.generated.json`, which is built by scanning
 * report files. This map only decides how each one looks; it deliberately has
 * no opinion about which state an experiment is in, because the previous
 * hand-written version of this table went stale three separate times in one
 * working session.
 */
function experimentTone(status: string): 'ready' | 'blocked' | 'not-trained' {
  if (status.includes('BLOCKED') && !status.includes('MEASURED')) return 'not-trained';
  if (status.startsWith('EXECUTED')) return 'ready';
  return 'blocked';
}

function ExperimentCard({ experiment }: { experiment: RegisteredExperiment }) {
  const tone = experimentTone(experiment.status);
  const blocker =
    typeof experiment.blocker === 'string'
      ? experiment.blocker
      : (experiment.blocker?.missing ?? null);

  return (
    <div className={`contrib ${tone}`}>
      <div className="contrib-head">
        <code>{experiment.id}</code>
        <span className={`status-pill ${tone}`}>{experiment.status}</span>
      </div>
      <div className="contrib-what">{experiment.title}</div>

      {experiment.hypothesis && (
        <p className="muted" style={{ marginTop: 8 }}>
          {experiment.hypothesis}
        </p>
      )}

      {experiment.arms_executed && experiment.arms_executed.length > 0 && (
        <table className="arm-table">
          <thead>
            <tr>
              <th>arm</th>
              <th className="num">bands</th>
              <th className="num">IoU</th>
              <th className="num">F1</th>
              <th className="num">prec</th>
              <th className="num">recall</th>
            </tr>
          </thead>
          <tbody>
            {experiment.arms_executed.map((arm) => (
              <tr key={arm.arm}>
                <td>
                  <code>{arm.arm}</code>
                </td>
                <td className="num">{arm.bands}</td>
                <td className="num">{arm.iou.toFixed(4)}</td>
                <td className="num">{arm.f1.toFixed(4)}</td>
                <td className="num">{arm.precision.toFixed(4)}</td>
                <td className="num">{arm.recall.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {typeof experiment.loro_iou === 'number' && (
        <dl className="kv" style={{ marginTop: 8 }}>
          <dt>Region-disjoint (LORO)</dt>
          <dd>{experiment.loro_iou.toFixed(4)} IoU</dd>
          {typeof experiment.official_iou === 'number' && (
            <>
              <dt>Official chip-level split</dt>
              <dd>{experiment.official_iou.toFixed(4)} IoU</dd>
              <dt>Gap</dt>
              <dd>
                {experiment.gap_iou !== undefined && experiment.gap_iou > 0 ? '+' : ''}
                {experiment.gap_iou?.toFixed(4)} IoU
              </dd>
            </>
          )}
        </dl>
      )}

      {experiment.finding && <p className="contrib-finding">{experiment.finding}</p>}

      {experiment.what_this_is_not && (
        <p className="analysis-warning">{experiment.what_this_is_not}</p>
      )}

      {blocker && (
        <p className="muted" style={{ marginTop: 8 }}>
          <strong>Blocked on: </strong>
          {blocker}
        </p>
      )}

      {experiment.arms_blocked && Object.keys(experiment.arms_blocked).length > 0 && (
        <ul className="analysis-bands" style={{ marginTop: 8 }}>
          {Object.entries(experiment.arms_blocked).map(([arm, why]) => (
            <li key={arm}>
              <code>{arm}</code> — {why}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Research({ experiments }: { experiments: Experiment[] }) {
  const complete = experiments.filter((e) => e.status === 'complete');
  const pending = experiments.filter((e) => e.status !== 'complete');

  return (
    <>
      <h2>Research &amp; Methodology</h2>
      <p className="lede">
        A Phase 0 gap analysis against seven papers established that the components SAT-AI
        assembles are individually well established. The contribution is therefore located in{' '}
        <strong>measurement</strong>, not architecture.
      </p>

      {/* Status, dataset, model and the reason each one is where it is. A
          contribution that has not run says so and says what is blocking it —
          the alternative is a table of four rows that all look equally ready. */}
      <div className="panel">
        <div className="panel-title">Contributions C1–C4</div>
        <div className="contrib-grid">
          {EXPERIMENTS_REGISTER.map((experiment) => (
            <ExperimentCard key={experiment.id} experiment={experiment} />
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-title">What the corpus establishes — and SAT-AI does not claim</div>
        <List items={ESTABLISHED} />
        <p className="muted" style={{ marginTop: 10 }}>
          A further finding: <strong>Sen1Floods11&apos;s official splits are not region-disjoint.</strong>{' '}
          Every region except Bolivia appears in train, validation and test at ~58/21/21, so chips
          from one flood event straddle the boundary. SAT-AI builds its own leave-one-region-out
          splits and reports both, so the inflation the standard protocol carries becomes a
          measured quantity.
        </p>
      </div>

      <h3>Executed experiments</h3>
      {complete.length === 0 && <div className="panel muted">Loading…</div>}
      {complete.map((e) => (
        <div className="panel" key={e.id}>
          <div className="pill-row" style={{ marginBottom: 8 }}>
            <span className="badge role">{e.contribution}</span>
            <strong style={{ fontSize: 15 }}>{e.name}</strong>
          </div>
          <div className="muted" style={{ marginBottom: 10 }}>
            {e.dataset} · {e.model} · completed {e.completed_at?.slice(0, 10)}
          </div>
          <pre>{JSON.stringify(e.metrics, null, 2)}</pre>
          <List items={e.notes} />
        </div>
      ))}

      <h3>Pending experiments</h3>
      <div className="panel">
        <p className="muted" style={{ marginBottom: 12 }}>
          Built and runnable but not executed. Their metrics are empty rather than estimated — a
          fabricated result would undo the entire argument of this project.
        </p>
        <table>
          <thead><tr><th>Experiment</th><th>C</th><th>Blocker</th></tr></thead>
          <tbody>{pending.map((e) => (
            <tr key={e.id}><td>{e.name}</td><td>{e.contribution}</td>
              <td className="muted">{e.notes[0] ?? '—'}</td></tr>))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// --- page ---

export default function Page() {
  const [section, setSection] = useState('overview');
  const [regions, setRegions] = useState<Region[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [region, setRegion] = useState('bihar_ganga');
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getRegions().then(setRegions).catch((e) => setErr(e.message));
    getExperiments().then(setExperiments).catch(() => undefined);
    getHealth().then(setHealth);
  }, []);

  // Study areas come from the generated catalogue, not from the database. The
  // catalogue is configuration and carries all seven regions; the database only
  // holds the four that were seeded, and results for any of them. Sourcing the
  // selector from the database would silently hide Nepal, Odisha and
  // Uttarakhand because no row exists for them yet.
  const area: StudyArea | undefined = STUDY_AREAS.find((a) => a.id === region);
  const done = experiments.filter((e) => e.status === 'complete').length;
  // Counted from artifacts on disk, not declared. The tile used to read a
  // hardcoded zero, which stayed zero after the first analysis was produced.
  const producedCount = ANALYSED_AREAS.reduce((total, a) => total + a.analyses.length, 0);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <h1>SAT-AI</h1>
          <div className="tag">Multi-hazard risk · research prototype</div>
        </div>
        {SECTIONS.map((g) => (
          <div key={g.group}>
            <div className="nav-group">{g.group}</div>
            {g.items.map(([id, label]) => (
              <button key={id} className={`nav-item ${section === id ? 'active' : ''}`}
                onClick={() => setSection(id)}>{label}</button>
            ))}
          </div>
        ))}
        <div style={{ padding: '18px 20px 0', fontSize: 11, color: 'var(--fg-faint)' }}>
          <div><span className={`status-dot ${health ? 'ok' : 'warn'}`} />
            API {health ? String((health as { status: string }).status) : 'not configured'}</div>
          <div style={{ marginTop: 4 }}><span className="status-dot ok" />Database connected</div>
        </div>
      </aside>

      <main className="main">
        {err && <div className="disclaimer">Data layer error: {err}</div>}

        {/* The hero leads the overview; on every other section the selector is
            the first thing, because the reader has already arrived and is
            choosing what to look at. */}
        {section === 'overview' && (
          <div className="hero">
            <div>
              <h1 className="hero-title">
                AI-driven <span className="accent">multi-hazard intelligence</span> from
                satellite observation
              </h1>
              <p className="hero-sub">
                Evidence-grounded satellite analysis for flood, wildfire, extreme-weather and
                post-event damage. Every value carries the model that produced it, the scenes
                behind it and how old the observation is — and where nothing has been computed,
                the interface says so rather than showing a number.
              </p>
              <div className="hero-tags">
                <span className="hero-tag">Sentinel-1 SAR</span>
                <span className="hero-tag">Terrain</span>
                <span className="hero-tag">Rainfall</span>
                <span className="hero-tag">Risk engine</span>
                <span className="hero-tag">Explainability</span>
                <span className="hero-tag">Grounded agents</span>
              </div>
            </div>
          </div>
        )}

        {!['research', 'about'].includes(section) && (
          <RegionSelector areas={STUDY_AREAS} selected={region} onSelect={setRegion} />
        )}

        {section === 'overview' && (
          <>
            <div className="cmd-map-wrap" style={{ position: 'relative' }}>
              <CommandMap areas={STUDY_AREAS} selectedId={region} onSelect={setRegion} />
              <AnalysisPanel area={area} result={null} />
            </div>

            <h3 style={{ marginTop: 22 }}>Analysis pipeline</h3>
            <p className="muted" style={{ maxWidth: '72ch', marginBottom: 4 }}>
              Stage state is derived from what actually exists, not decoration. This is the fastest
              way to see that the system is wired end to end and where it currently stops.
            </p>
            <PipelineStrip stages={pipelineFor(area, null)} />

            <Disclaimer />

            <div className="grid grid-2" style={{ marginTop: 16 }}>
              <ModelStatus
                name="Flood U-Net (VV, VH, VV/VH ratio)"
                version="1.0.0 · loro_india"
                dataset="Sen1Floods11 v1.1 HandLabeled — 446 chips, 11 flood events"
                split="Leave-one-region-out (ADR-009); India held out entirely"
                testRegion="India, 68 chips — influenced neither training nor selection"
                status="READY"
                metrics={[
                  ['IoU (India, test)', '0.523'],
                  ['F1', '0.687'],
                  ['Precision', '0.751'],
                  ['Recall', '0.633'],
                  ['vs Otsu baseline', '+0.148 IoU'],
                  ['Per-chip IoU median', '0.223'],
                  ['Parameters', '7.76 M'],
                ]}
                blocker="Validation IoU on the fold's validation region (Mekong) was 0.868; the test score on India is 0.523. That gap is what quoting the validation number would have overstated it by. Pooled IoU is also dominated by chips with large water bodies — the per-chip median is 0.223, so the model does well where there is a lot of water and poorly where there is little."
              />
              <ModelStatus
                name="Otsu classical baseline (VV, no HAND mask)"
                version="1.1.0"
                dataset="Sen1Floods11 v1.1 HandLabeled — 441 of 446 chips scored"
                split="Leave-one-region-out guard selection, 11 regions"
                testRegion="Each region held out in turn; counts pooled"
                status="READY"
                metrics={[
                  ['IoU (pooled, region-disjoint)', '0.420'],
                  ['F1', '0.591'],
                  ['Precision', '0.781'],
                  ['Recall', '0.476'],
                  ['India (68 chips)', 'IoU 0.375'],
                  ['Best region — Mekong', 'IoU 0.790'],
                  ['Worst region — Somalia', 'IoU 0.000'],
                ]}
                blocker="Sen1Floods11 ships no HAND raster, so the terrain mask that removes Otsu's false positives over tarmac and dry sand is absent. These are a LOWER BOUND on the method as operationally deployed. Any deep model must beat 0.420 on this protocol to have demonstrated anything."
              />
            </div>

            <ScopePanel scope={SCOPE_OVERALL} title="Scope" />

            <div className="grid grid-2">
              <div className="panel">
                <div className="panel-title">Research register</div>
                <table>
                  <tbody>
                    <tr><td>Study areas configured</td><td className="num">{STUDY_AREAS.length}</td></tr>
                    <tr><td>Countries</td><td className="num">
                      {new Set(STUDY_AREAS.map((a) => a.country)).size}
                    </td></tr>
                    <tr><td>Verified historical events</td><td className="num">
                      {areasWithVerifiedEvents().reduce((n, a) => n + verifiedEvents(a).length, 0)}
                    </td></tr>
                    <tr><td>Experiments registered</td><td className="num">{EXPERIMENTS_REGISTER.length}</td></tr>
                    <tr><td>Experiments executed</td><td className="num">{executedExperiments().length}</td></tr>
                    <tr><td>Analyses produced</td><td className="num">{producedCount}</td></tr>
                    <tr><td>Catalogue version</td><td className="num">{CATALOGUE_VERSION}</td></tr>
                  </tbody>
                </table>
              </div>
              <SatellitePanel event={area ? verifiedEvents(area)[0] : undefined} />
              <AnalysesPanel area={area} />
              <ExplainabilityPanel />
            </div>
          </>
        )}

        {section === 'map' && (
          <>
            <h2>Risk Map</h2>
            <p className="lede">
              Configured study areas across India and Nepal. Hazard and risk layers appear once the
              inference pipeline has produced them; the layer control lists every layer with the
              reason it is unavailable rather than hiding it.
            </p>
            <Disclaimer />
            <div style={{ position: 'relative' }}>
              <CommandMap areas={STUDY_AREAS} selectedId={region} onSelect={setRegion} />
              <AnalysisPanel area={area} result={null} />
            </div>
            {area && (
              <div className="grid grid-2" style={{ marginTop: 16 }}>
                <div className="panel">
                  <div className="panel-title">Why this region is in the study</div>
                  <p className="muted">{area.selectionRationale}</p>
                  <table style={{ marginTop: 10 }}>
                    <tbody>
                      <tr><td>Role</td><td>{ROLE_LABEL[area.studyRole].label}</td></tr>
                      <tr><td>Bounding box</td><td className="num">{area.bbox.join(', ')}</td></tr>
                      <tr><td>Analysis CRS</td><td className="num">{area.utmEpsg}</td></tr>
                      <tr><td>Area</td><td className="num">{area.areaKm2.toLocaleString()} km²</td></tr>
                      <tr><td>Tiles (512 px @ 10 m)</td><td className="num">{area.tileCount}</td></tr>
                      <tr><td>Ground truth</td><td className="muted">
                        {area.labelSources[0] ?? 'none — transfer target only'}
                      </td></tr>
                    </tbody>
                  </table>
                  {area.notes && <p className="panel-note">{area.notes}</p>}
                </div>
                <SatellitePanel event={verifiedEvents(area)[0]} />
              </div>
            )}
          </>
        )}

        {['flood', 'wildfire', 'cyclone', 'damage'].includes(section) && (
          <HazardSection hazard={section} region={region} />
        )}

        {section === 'xai' && (
          <>
            <h2>Explainable AI</h2>
            <p className="lede">
              Different model classes need different tools. Tree ensembles get TreeSHAP;
              segmentation networks get per-modality occlusion and integrated gradients, because
              Grad-CAM on a U-Net is close to meaningless — segmentation output is already
              spatially localised.
            </p>
            <div className="panel">
              <div className="panel-title">Method by model class</div>
              <table>
                <thead><tr><th>Model</th><th>Method</th><th>Output</th></tr></thead>
                <tbody>{XAI_METHODS.map(([m, meth, out]) => (
                  <tr key={m}><td>{m}</td><td>{meth}</td><td className="muted">{out}</td></tr>))}
                </tbody>
              </table>
            </div>
            <div className="unavailable">
              <div className="h">No explanation artifacts yet</div>
              <p className="r">
                Explanations are computed in the batch plane alongside each prediction, not on
                request — SHAP on demand would blow the API latency budget. No predictions have been
                produced, so there are no attributions to show.
              </p>
              <div className="w">ml/flood/explain.py, after a completed inference run</div>
            </div>
          </>
        )}

        {section === 'warning' && (
          <>
            <h2>Early Warning</h2>
            <p className="lede">
              SAT-AI generates structured risk alerts for research purposes. The language it uses is
              deliberately constrained: it reports model-estimated risk, never that a hazard will occur.
            </p>
            <Disclaimer />
            <div className="panel">
              <div className="panel-title">Alert language contract</div>
              <div className="grid grid-2">
                <div><h3 style={{ marginTop: 0, color: 'var(--obs)' }}>Permitted</h3><List items={ALERT_PERMITTED} /></div>
                <div><h3 style={{ marginTop: 0, color: 'var(--danger)' }}>Never emitted</h3><List items={ALERT_FORBIDDEN} /></div>
              </div>
            </div>
            <div className="unavailable">
              <div className="h">No alerts</div>
              <p className="r">No hazard analyses exist, so no alert objects have been generated.</p>
              <div className="w">Alerts are emitted by the risk engine after an inference run.</div>
            </div>
          </>
        )}

        {section === 'historical' && (
          <>
            <h2>Historical Analysis</h2>
            <p className="lede">
              Time series of previous SAT-AI analyses for a region. These are past model outputs, not
              independent observations of what occurred — an important distinction when reading a trend.
            </p>
            <div className="unavailable">
              <div className="h">No history</div>
              <p className="r">
                The result archive is empty because no analyses have been run. Once the pipeline
                executes on a schedule, this section shows the risk-index series with each point&apos;s
                model version and scene provenance.
              </p>
              <div className="w">GET /api/v1/historical?region={region}&amp;hazard=flood&amp;months=12</div>
            </div>
          </>
        )}

        {section === 'assistant' && (
          <Assistant
            regions={regions}
            region={region}
            onMapAction={(regionId) => {
              // Switch the dashboard's region so the map reflects what the
              // agent just talked about, then show it.
              setRegion(regionId);
              setSection('map');
            }}
          />
        )}
        {section === 'research' && <Research experiments={experiments} />}

        {section === 'about' && (
          <>
            <h2>About &amp; Limitations</h2>
            <p className="lede">
              SAT-AI is a student research prototype built for a B.Tech project at Vidyalankar
              Institute of Technology, Mumbai. The limitations below are the most scientifically
              important part of the system; the full living document is <code>docs/limitations.md</code>.
            </p>
            <Disclaimer />
            <div className="panel">
              <div className="panel-title">Permanent scope boundaries — not deficiencies to be fixed</div>
              <List items={BOUNDARIES} />
            </div>
            <div className="panel">
              <div className="panel-title">Latency, measured per stream</div>
              <table>
                <thead><tr><th>Stream</th><th>Class</th><th>Approximate</th></tr></thead>
                <tbody>{LATENCY.map(([s, c, v]) => (
                  <tr key={s}><td>{s}</td><td>{c}</td><td className="num">{v}</td></tr>))}
                </tbody>
              </table>
            </div>
            <div className="panel">
              <div className="panel-title">Known methodological limitations</div>
              <List items={LIMITATIONS} />
            </div>
            <div className="panel">
              <div className="panel-title">Colophon</div>
              <p className="muted">
                Next.js on Vercel · FastAPI serving plane · PostgreSQL + PostGIS · Google Gemini
                with native function calling, over a deterministic router that needs no model.
                Data: Copernicus Sentinel-1/2, Copernicus DEM,
                NASA GPM IMERG, ERA5, NASA FIRMS, IBTrACS, WorldPop, GHSL, OpenStreetMap, Sen1Floods11.
              </p>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
