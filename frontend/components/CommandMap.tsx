'use client';

/**
 * The map, as the centrepiece rather than an illustration.
 *
 * Two rules govern what it is allowed to draw:
 *
 * 1. A layer that has no data is listed and disabled, with the reason. It is
 *    never silently absent and never drawn empty. A user toggling FLOOD and
 *    seeing nothing cannot tell "no flooding" from "never computed", so the
 *    control says which.
 * 2. Study-area rectangles are the AOI bounding boxes, and the legend says so.
 *    They are not hazard extents, and drawing them in a hazard colour would
 *    make configuration look like a result.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import 'maplibre-gl/dist/maplibre-gl.css';
import type { StudyArea } from '@/lib/study-areas';
import { ROLE_LABEL, verifiedEvents } from '@/lib/study-areas';
import type { Analysis } from '@/lib/analyses';
import { ANALYSED_AREAS, WILDFIRE, analysesFor, mapOverlays } from '@/lib/analyses';

export type LayerId =
  | 'aoi'
  | 'flood'
  | 'wildfire'
  | 'cyclone'
  | 'rainfall'
  | 'terrain'
  | 'risk'
  | 'exposure'
  | 'vulnerability'
  | 'damage';

export interface LayerState {
  id: LayerId;
  label: string;
  /** Instrument or product behind the layer. */
  source: string;
  /** observation | model | derived | index — mirrors satai.provenance.SourceKind. */
  kind: string;
  /** Null when nothing has been computed; the control then disables the toggle. */
  timestamp: string | null;
  available: boolean;
  /** Why it is unavailable. Shown in the control so absence is explained. */
  reason?: string;
}

/**
 * Layer availability is derived from the analysis catalogue, never hardcoded.
 *
 * This function used to carry hand-written reasons, and every one of them went
 * stale the moment the thing it described was done. It claimed the flood model
 * "has not been trained" after it was trained and evaluated, and that FIRMS
 * "requires a FIRMS_MAP_KEY" after real detections had been fetched from the
 * keyless archive. Both statements were visible to users and both were false.
 *
 * So the reasons now come from `analyses.generated.json`, which is built from
 * artifacts on disk. A layer becomes available when an analysis exists **and**
 * passed its validation; an analysis that ran but failed validation reports
 * that specifically, because "not validated" and "not attempted" are different
 * things and only one of them is a gap in the work.
 */
export function buildLayers(area: StudyArea | undefined): LayerState[] {
  const events = area ? verifiedEvents(area) : [];
  const floodEvent = events.find((e) => e.hazard === 'flood');
  const produced = area ? analysesFor(area.id) : [];
  // A drawable analysis wins over one that was withheld: the Nepal scene has
  // both a gate-failed U-Net run and a valid baseline extent, and the layer is
  // available because of the second, not unavailable because of the first.
  const pick = (hazard: string, kind?: Analysis['kind']) => {
    const matches = produced.filter(
      (a) => a.hazard === hazard && (kind === undefined || a.kind === kind),
    );
    return matches.find((a) => a.displayable) ?? matches[0];
  };
  const hasOverlay = (layer: string) =>
    produced.some((a) => a.displayable && (a.overlays ?? []).some((o) => o.layer === layer));

  const flood = pick('flood', 'model_inference');
  const floodRisk = pick('flood', 'risk_analysis');
  const burn = produced.find((a) => a.hazard === 'wildfire' && a.kind === 'historical_analysis');
  const fire = produced.find((a) => a.hazard === 'wildfire' && a.kind === 'observation');
  const cyclone = pick('cyclone');
  const withheld = produced.find((a) => a.hazard === 'flood' && a.displayable === false);

  return [
    {
      id: 'aoi',
      label: 'Study areas',
      source: 'configs/aoi.yaml',
      kind: 'catalogue',
      timestamp: null,
      available: true,
    },
    {
      id: 'flood',
      label: 'Flood extent (hazard)',
      source: flood
        ? `${String(flood.model?.name ?? 'model')} on ${String(flood.observation?.scene_id ?? 'Sentinel-1').slice(0, 26)}`
        : floodEvent
          ? floodEvent.sensor
          : 'Sentinel-1 SAR',
      kind: 'model',
      timestamp: (flood?.processing?.processed_at as string) ?? null,
      available: Boolean(flood?.displayable && hasOverlay('flood')),
      reason: flood?.displayable
        ? undefined
        : (withheld?.withheld_reason ??
          (floodEvent
            ? `A verified Sentinel-1 event is on file for ${floodEvent.occurredOn}, but no flood map has been produced for this area.`
            : 'No flood map has been produced for this area.')),
    },
    {
      id: 'exposure',
      label: 'Population (exposure)',
      source: 'WorldPop 2020',
      kind: 'model',
      timestamp: null,
      available: hasOverlay('exposure'),
      reason: hasOverlay('exposure')
        ? undefined
        : 'Exposure is drawn where a risk analysis has been run; none has for this area.',
    },
    {
      id: 'vulnerability',
      label: 'Vulnerability',
      source: 'no source acquired',
      kind: 'index',
      timestamp: null,
      available: false,
      reason:
        (floodRisk?.vulnerability?.reason as string | undefined) ??
        'No vulnerability layer exists for any study area. The risk engine excludes the term rather than imputing it.',
    },
    {
      id: 'risk',
      label: 'Risk (combined)',
      source: 'SAT-AI risk engine, R = H^a E^b',
      kind: 'index',
      timestamp: null,
      available: hasOverlay('risk'),
      reason: hasOverlay('risk')
        ? undefined
        : 'Risk needs a validated hazard field and exposure for this area.',
    },
    {
      id: 'wildfire',
      label: burn ? 'Burn severity and active fire' : 'Active fire detections',
      source: burn ? 'Sentinel-2 dNBR, NASA FIRMS' : 'NASA FIRMS (VIIRS, keyless archive)',
      kind: burn ? 'derived' : 'observation',
      timestamp: WILDFIRE?.run_at ?? null,
      available: Boolean(burn?.displayable || fire?.detections?.length),
      reason:
        burn?.displayable || fire?.detections?.length
          ? undefined
          : `No burn-severity analysis for this area, and no detections in the last ${WILDFIRE?.window ?? '7d'}. No detection is not the same as no fire.`,
    },
    {
      id: 'cyclone',
      label: 'Cyclone track and wind',
      source: 'IBTrACS v04r01 best track',
      kind: 'observation',
      timestamp: (cyclone?.observation?.last_fix as string) ?? null,
      available: cyclone?.displayable === true,
      reason: cyclone ? undefined : 'No historical cyclone analysis has been run for this area.',
    },
    {
      id: 'rainfall',
      label: 'Rainfall',
      source: 'GPM IMERG',
      kind: 'observation',
      timestamp: null,
      available: false,
      reason:
        'IMERG is used as a model input for C2, not drawn: at 0.1 degrees it is one value per study-area chip.',
    },
    {
      id: 'terrain',
      label: 'Terrain',
      source: 'Copernicus DEM GLO-30',
      kind: 'derived',
      timestamp: null,
      available: false,
      reason: 'Terrain is used as a model input for C2 at chip scale, not drawn as a map layer.',
    },
    {
      id: 'damage',
      label: 'Post-event change',
      source: 'Sentinel-1 pre/post pair',
      kind: 'derived',
      timestamp: null,
      available: hasOverlay('damage'),
      reason: hasOverlay('damage')
        ? undefined
        : produced.some((a) => a.hazard === 'damage')
          ? 'Change was measured for this area; its full-resolution raster is summarised in the panel rather than drawn.'
          : 'Change detection needs a same-orbit pre/post pair around an event.',
    },
  ];
}

/**
 * The observed cyclone track, as a feature collection.
 *
 * Empty when no cyclone analysis has been produced, which is a real state and
 * not an error: the map then simply has no track on it.
 */
function cycloneTrackGeoJSON(): GeoJSON.FeatureCollection {
  const empty: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] };
  const features = ANALYSED_AREAS.flatMap((area) =>
    area.analyses
      .filter((a) => a.hazard === 'cyclone' && a.displayable)
      .flatMap((a) => (a.geometry?.features ?? []) as GeoJSON.Feature[]),
  );
  return features.length ? { type: 'FeatureCollection', features } : empty;
}

/** Active-fire detections across every study area that had any. */
function fireDetectionsGeoJSON(): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = ANALYSED_AREAS.flatMap((area) =>
    area.analyses
      .filter((a) => a.hazard === 'wildfire')
      .flatMap((a) =>
        (a.detections ?? []).map((d) => ({
          type: 'Feature' as const,
          geometry: { type: 'Point' as const, coordinates: [d.lon, d.lat] },
          properties: {
            observed_at: d.observed_at,
            frp_mw: d.frp_mw,
            confidence: d.confidence,
            aoi: area.id,
          },
        })),
      ),
  );
  return { type: 'FeatureCollection', features };
}

/**
 * A popup built as DOM rather than an HTML string.
 *
 * Every value here comes from a data file, and interpolating it into markup
 * would make the map's rendering path depend on that content being safe.
 */
function popup(
  maplibre: typeof import('maplibre-gl'),
  map: import('maplibre-gl').Map,
  lngLat: import('maplibre-gl').LngLatLike,
  rows: [string, string][],
): void {
  const container = document.createElement('div');
  container.className = 'map-popup';
  for (const [label, value] of rows) {
    const line = document.createElement('div');
    const strong = document.createElement('strong');
    strong.textContent = label;
    line.append(strong, document.createTextNode(value));
    container.append(line);
  }
  new maplibre.Popup({ closeButton: true })
    .setLngLat(lngLat)
    .setDOMContent(container)
    .addTo(map);
}

const ROLE_COLOUR: Record<string, string> = {
  training: '#38bdf8',
  transfer_evaluation: '#a78bfa',
  candidate: '#64748b',
};

export function CommandMap({
  areas,
  selectedId,
  onSelect,
}: {
  areas: StudyArea[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<import('maplibre-gl').Map | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [tilesFailed, setTilesFailed] = useState(false);
  const [layers, setLayers] = useState<LayerState[]>(() => buildLayers(undefined));
  const [open, setOpen] = useState(true);

  const selected = areas.find((a) => a.id === selectedId);

  useEffect(() => setLayers(buildLayers(selected)), [selected]);

  useEffect(() => {
    const node = container.current;
    if (!node || areas.length === 0) return;

    // The cleanup runs synchronously while the map is only assigned after the
    // dynamic import resolves. Without this flag, unmounting inside that window
    // (React StrictMode does it on every mount in development) leaves a live map
    // holding a WebGL context on a detached node.
    let cancelled = false;

    (async () => {
      try {
        const maplibre = (await import('maplibre-gl')).default;
        if (cancelled) return;

        const map = new maplibre.Map({
          container: node,
          style: {
            version: 8,
            sources: {
              carto: {
                type: 'raster',
                tiles: [
                  'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
                  'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
                ],
                tileSize: 256,
                attribution: '© OpenStreetMap contributors © CARTO',
              },
            },
            layers: [{ id: 'carto', type: 'raster', source: 'carto' }],
          },
          center: [85.5, 24.5],
          zoom: 4.2,
          attributionControl: false,
        });
        mapRef.current = map;

        map.addControl(new maplibre.NavigationControl({ showCompass: false }), 'bottom-right');
        map.addControl(new maplibre.AttributionControl({ compact: true }), 'bottom-left');
        map.addControl(new maplibre.ScaleControl({ maxWidth: 110, unit: 'metric' }), 'bottom-left');

        // MapLibre reports tile failures through this event and otherwise just
        // renders an empty canvas. A basemap that silently fails to load looks
        // identical to a map with nothing on it, which is exactly the ambiguity
        // this interface is supposed to eliminate — so it is surfaced.
        map.on('error', (e) => {
          const message = (e as { error?: { message?: string } }).error?.message ?? '';
          if (message.includes('tile') || message.includes('Failed to fetch')) {
            if (!cancelled) setTilesFailed(true);
          }
        });

        map.on('load', () => {
          if (cancelled) return;
          map.addSource('aoi', {
            type: 'geojson',
            data: {
              type: 'FeatureCollection',
              features: areas.map((a) => ({
                type: 'Feature' as const,
                id: a.id,
                properties: {
                  id: a.id,
                  name: a.name,
                  role: a.studyRole,
                  colour: ROLE_COLOUR[a.studyRole] ?? ROLE_COLOUR.candidate,
                  verified: verifiedEvents(a).length,
                },
                geometry: {
                  type: 'Polygon' as const,
                  coordinates: [
                    [
                      [a.bbox[0], a.bbox[1]],
                      [a.bbox[2], a.bbox[1]],
                      [a.bbox[2], a.bbox[3]],
                      [a.bbox[0], a.bbox[3]],
                      [a.bbox[0], a.bbox[1]],
                    ],
                  ],
                },
              })),
            },
          });

          map.addLayer({
            id: 'aoi-fill',
            type: 'fill',
            source: 'aoi',
            paint: {
              'fill-color': ['get', 'colour'],
              'fill-opacity': ['case', ['boolean', ['feature-state', 'selected'], false], 0.3, 0.1],
            },
          });
          map.addLayer({
            id: 'aoi-line',
            type: 'line',
            source: 'aoi',
            paint: {
              'line-color': ['get', 'colour'],
              'line-width': ['case', ['boolean', ['feature-state', 'selected'], false], 2.4, 1.1],
            },
          });
          // No symbol/text layer here on purpose. A `text-field` requires the
          // style to declare a `glyphs` URL, and this style is raster-only —
          // adding one throws inside the load handler and silently aborts every
          // line after it, which is how the whole overlay went missing once.
          // Region names are carried by the selector and the analysis panel,
          // and seven labelled rectangles at country zoom would overlap anyway.

          // --- real analyses, drawn only where one exists ------------------
          //
          // Both sources are added unconditionally with whatever geometry the
          // catalogue holds, and are empty collections when nothing has been
          // produced. Adding a source conditionally inside a load handler is
          // how a layer ends up referenced before it exists.

          // Raster results, drawn from their small lon/lat PNGs. Added before
          // the point and line layers so a track or a detection is never hidden
          // under an overlay. Only validated analyses reach this list.
          mapOverlays().forEach((overlay, index) => {
            const id = `overlay-${overlay.layer}-${index}`;
            map.addSource(id, {
              type: 'image',
              url: overlay.url,
              coordinates: overlay.coordinates as [
                [number, number],
                [number, number],
                [number, number],
                [number, number],
              ],
            });
            map.addLayer({
              id: `${id}-layer`,
              type: 'raster',
              source: id,
              paint: { 'raster-opacity': 0.85, 'raster-fade-duration': 0 },
            });
            registerMapLayer(overlay.layer, `${id}-layer`);
          });

          map.addSource('cyclone-track', {
            type: 'geojson',
            data: cycloneTrackGeoJSON(),
          });
          map.addLayer({
            id: 'cyclone-track-line',
            type: 'line',
            source: 'cyclone-track',
            filter: ['==', ['geometry-type'], 'LineString'],
            paint: {
              'line-color': '#f472b6',
              'line-width': 2,
              'line-opacity': 0.9,
            },
          });
          map.addLayer({
            id: 'cyclone-track-fixes',
            type: 'circle',
            source: 'cyclone-track',
            filter: ['==', ['geometry-type'], 'Point'],
            paint: {
              // Radius carries the observed wind, so the track reads as a
              // storm rather than a line. Interpolated on the real m/s range.
              'circle-radius': [
                'interpolate',
                ['linear'],
                ['coalesce', ['get', 'max_wind_ms'], 0],
                10, 2.5,
                60, 8,
              ],
              'circle-color': '#f472b6',
              'circle-opacity': 0.55,
              'circle-stroke-color': '#fff',
              'circle-stroke-width': 0.5,
            },
          });

          map.addSource('fire-detections', {
            type: 'geojson',
            data: fireDetectionsGeoJSON(),
          });
          map.addLayer({
            id: 'fire-detection-points',
            type: 'circle',
            source: 'fire-detections',
            paint: {
              'circle-radius': 5,
              'circle-color': '#fb923c',
              'circle-opacity': 0.85,
              'circle-stroke-color': '#7c2d12',
              'circle-stroke-width': 1,
            },
          });

          registerMapLayer('cyclone', 'cyclone-track-line');
          registerMapLayer('cyclone', 'cyclone-track-fixes');
          registerMapLayer('wildfire', 'fire-detection-points');
          registerMapLayer('aoi', 'aoi-fill');
          registerMapLayer('aoi', 'aoi-line');
          applyVisibility(map, visibleRef.current);

          map.on('click', 'cyclone-track-fixes', (e) => {
            const props = e.features?.[0]?.properties ?? {};
            popup(maplibre, map, e.lngLat, [
              ['Observed fix', String(props.observed_at ?? '')],
              ['Max wind', `${String(props.max_wind_ms ?? '?')} m/s`],
              ['Source', 'IBTrACS v04r01 best track (OBSERVATION)'],
            ]);
          });
          map.on('click', 'fire-detection-points', (e) => {
            const props = e.features?.[0]?.properties ?? {};
            popup(maplibre, map, e.lngLat, [
              ['Detected', String(props.observed_at ?? '')],
              ['Radiative power', `${String(props.frp_mw ?? '?')} MW`],
              ['Confidence', String(props.confidence ?? 'unknown')],
              ['Source', 'NASA FIRMS VIIRS (OBSERVATION)'],
            ]);
          });

          map.on('click', 'aoi-fill', (e) => {
            const id = e.features?.[0]?.properties?.id;
            if (typeof id === 'string') onSelect(id);
          });
          map.on('mouseenter', 'aoi-fill', () => {
            map.getCanvas().style.cursor = 'pointer';
          });
          map.on('mouseleave', 'aoi-fill', () => {
            map.getCanvas().style.cursor = '';
          });

          setReady(true);
        });
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      }
    })();

    return () => {
      cancelled = true;
      mapRef.current?.remove();
      mapRef.current = null;
      setReady(false);
    };
  }, [areas, onSelect]);

  // Selection drives both the highlight and the camera. Split from the setup
  // effect so changing region does not tear down and rebuild the map.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;

    for (const area of areas) {
      map.setFeatureState({ source: 'aoi', id: area.id }, { selected: area.id === selectedId });
    }
    const area = areas.find((a) => a.id === selectedId);
    if (area) {
      map.fitBounds(
        [
          [area.bbox[0], area.bbox[1]],
          [area.bbox[2], area.bbox[3]],
        ],
        { padding: { top: 70, bottom: 70, left: 330, right: 70 }, duration: 900, maxZoom: 9 }
      );
    }
  }, [selectedId, areas, ready]);

  // Which map layers belong to which control. Filled while the map loads,
  // because overlay ids depend on what the catalogue holds.
  const groupsRef = useRef<Map<LayerId, string[]>>(new Map());
  // Everything starts visible. The previous control only ever showed "Study
  // areas" as checked while every other layer was drawn anyway, and its toggle
  // returned the list unchanged, so the checkboxes did nothing at all.
  const [visible, setVisible] = useState<Set<LayerId>>(
    () => new Set<LayerId>(['aoi', 'flood', 'wildfire', 'cyclone', 'risk', 'exposure', 'damage']),
  );
  const visibleRef = useRef(visible);
  visibleRef.current = visible;

  function registerMapLayer(group: LayerId, mapLayerId: string): void {
    const ids = groupsRef.current.get(group) ?? [];
    if (!ids.includes(mapLayerId)) ids.push(mapLayerId);
    groupsRef.current.set(group, ids);
  }

  function applyVisibility(map: import('maplibre-gl').Map, shown: Set<LayerId>): void {
    groupsRef.current.forEach((ids, group) => {
      for (const id of ids) {
        if (map.getLayer(id)) {
          map.setLayoutProperty(id, 'visibility', shown.has(group) ? 'visible' : 'none');
        }
      }
    });
  }

  const toggle = useCallback((id: LayerId) => {
    setVisible((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      const map = mapRef.current;
      if (map) applyVisibility(map, next);
      return next;
    });
    // registerMapLayer/applyVisibility read refs only, so they are stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (error) {
    return (
      <div className="unavailable">
        <div className="h">Map unavailable</div>
        <p className="r">{error}</p>
        <div className="w">The rest of the dashboard is unaffected.</div>
      </div>
    );
  }

  return (
    <div className="cmd-map">
      <div ref={container} className="cmd-map-canvas" />

      {tilesFailed && (
        <div className="cmd-tile-warning">
          Basemap tiles could not be loaded. Study-area outlines are still drawn from local
          configuration; the missing background is a network problem, not an absence of data.
        </div>
      )}

      <div className={`cmd-layers ${open ? '' : 'collapsed'}`}>
        <button className="cmd-layers-head" onClick={() => setOpen((v) => !v)} type="button">
          <span>Layers</span>
          <span className="cmd-chev">{open ? '−' : '+'}</span>
        </button>
        {open && (
          <div className="cmd-layers-body">
            {layers.map((layer) => (
              <div
                key={layer.id}
                className={`cmd-layer ${layer.available ? '' : 'disabled'}`}
                title={layer.available ? layer.source : layer.reason}
              >
                <label className="cmd-layer-row">
                  <input
                    type="checkbox"
                    checked={layer.available && visible.has(layer.id)}
                    disabled={!layer.available}
                    onChange={() => toggle(layer.id)}
                  />
                  <span className="cmd-layer-name">{layer.label}</span>
                  <span className={`kind-dot ${layer.kind}`} aria-hidden="true" />
                </label>
                <div className="cmd-layer-meta">
                  <span>{layer.source}</span>
                  {!layer.available && <span className="cmd-layer-off">no data</span>}
                </div>
                {!layer.available && layer.reason && (
                  <p className="cmd-layer-reason">{layer.reason}</p>
                )}
              </div>
            ))}
            <div className="cmd-legend">
              <div className="cmd-legend-title">Study area outlines</div>
              {(['training', 'transfer_evaluation', 'candidate'] as const).map((role) => (
                <div key={role} className="cmd-legend-row">
                  <span className="swatch" style={{ background: ROLE_COLOUR[role] }} />
                  {ROLE_LABEL[role].label}
                </div>
              ))}
              <p className="cmd-legend-note">
                Rectangles are AOI bounding boxes from the configuration. They are not hazard
                extents.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
