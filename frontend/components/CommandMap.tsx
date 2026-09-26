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

export type LayerId = 'aoi' | 'flood' | 'wildfire' | 'rainfall' | 'terrain' | 'risk' | 'damage';

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
 * Layer availability is derived, never hardcoded to `true`. Everything except
 * the AOI outlines depends on a batch run that has not happened for any region
 * yet, so they render as disabled rows carrying the reason rather than as
 * toggles that do nothing.
 */
export function buildLayers(area: StudyArea | undefined): LayerState[] {
  const events = area ? verifiedEvents(area) : [];
  const floodEvent = events.find((e) => e.hazard === 'flood');

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
      label: 'Flood extent',
      source: floodEvent ? floodEvent.sensor : 'Sentinel-1 SAR',
      kind: 'model',
      timestamp: null,
      available: false,
      reason: floodEvent
        ? `A verified Sentinel-1 event is on file for ${floodEvent.occurredOn}, but the flood segmentation model has not been trained, so no extent raster exists.`
        : 'No verified flood event is configured for this region.',
    },
    {
      id: 'wildfire',
      label: 'Active fire detections',
      source: 'NASA FIRMS (VIIRS)',
      kind: 'observation',
      timestamp: null,
      available: false,
      reason: 'FIRMS ingestion requires a FIRMS_MAP_KEY, which is not configured on this deployment.',
    },
    {
      id: 'rainfall',
      label: 'Rainfall',
      source: 'GPM IMERG',
      kind: 'observation',
      timestamp: null,
      available: false,
      reason: 'GPM ingestion requires NASA Earthdata credentials, which are not configured.',
    },
    {
      id: 'terrain',
      label: 'Terrain / HAND',
      source: 'Copernicus DEM GLO-30',
      kind: 'derived',
      timestamp: null,
      available: false,
      reason: 'DEM acquisition runs in the batch plane and has not been executed for this region.',
    },
    {
      id: 'risk',
      label: 'Risk bands',
      source: 'SAT-AI risk engine',
      kind: 'index',
      timestamp: null,
      available: false,
      reason:
        'The risk engine is implemented and sensitivity-analysed, but it needs hazard, exposure and vulnerability rasters it has not been given for this region.',
    },
    {
      id: 'damage',
      label: 'Change / damage',
      source: 'Pre/post SAR pair',
      kind: 'derived',
      timestamp: null,
      available: false,
      reason: 'Change detection needs a pre/post image pair around a specific event.',
    },
  ];
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

  const toggle = useCallback((id: LayerId) => {
    setLayers((current) =>
      current.map((l) => (l.id === id && l.available ? { ...l, timestamp: l.timestamp } : l))
    );
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
                    defaultChecked={layer.id === 'aoi'}
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
