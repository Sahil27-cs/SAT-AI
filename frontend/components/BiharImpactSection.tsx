'use client';

import { useState } from 'react';
import BIHAR_DATA from '@/lib/bihar-impact-data.json';

type SortKey = 'flooded_km2' | 'flood_pct' | 'impact_index' | 'cropland_km2';

export function BiharImpactSection() {
  const [selectedDate, setSelectedDate] = useState('2022-10-15');
  const [sortKey, setSortKey] = useState<SortKey>('flooded_km2');
  const [selectedDistrict, setSelectedDistrict] = useState<string>('Muzaffarpur');
  const [activeLayers, setActiveLayers] = useState({
    floodExtent: true,
    districtBoundaries: true,
    cropland: true,
    buildings: false,
    roads: false,
    historicalHazard: true,
  });
  const [evidenceOpen, setEvidenceOpen] = useState(false);

  const eventData =
    BIHAR_DATA.events.find((e) => e.date === selectedDate) || BIHAR_DATA.events[0];

  const sortedDistricts = [...BIHAR_DATA.districts].sort((a, b) => b[sortKey] - a[sortKey]);

  const activeDistrictData =
    BIHAR_DATA.districts.find((d) => d.name === selectedDistrict) || BIHAR_DATA.districts[0];

  const toggleLayer = (layer: keyof typeof activeLayers) => {
    setActiveLayers((prev) => ({ ...prev, [layer]: !prev[layer] }));
  };

  return (
    <section id="bihar-impact" className="section-padding">
      <div className="section-header">
        <div className="section-tag">Geospatial Intelligence Engine</div>
        <h2 className="section-title">
          Bihar Flood <span className="gradient-text">Impact Assessment</span>
        </h2>
        <p className="section-lead">
          Evidence-based flood inundation, cropland damage, infrastructure exposure, and historical hazard zonation.
        </p>
      </div>

      {/* Date Selector & Disclaimer Bar */}
      <div className="impact-control-bar">
        <div className="event-date-selector">
          <span className="control-label">Satellite Observation Event:</span>
          {BIHAR_DATA.events.map((e) => {
            const isRecent = e.date === '2024-07-28';
            return (
              <button
                key={e.date}
                type="button"
                className={`event-date-btn ${selectedDate === e.date ? 'active' : ''}`}
                onClick={() => setSelectedDate(e.date)}
              >
                📅 {e.date} — {e.title.split('(')[0].trim()}
                {isRecent ? (
                  <span style={{ marginLeft: 8, fontSize: '10px', background: 'rgba(56, 189, 248, 0.2)', color: '#38bdf8', padding: '2px 6px', borderRadius: 4, fontWeight: 700 }}>
                    RECENT OBSERVATION
                  </span>
                ) : (
                  <span style={{ marginLeft: 8, fontSize: '10px', background: 'rgba(148, 163, 184, 0.15)', color: '#94a3b8', padding: '2px 6px', borderRadius: 4 }}>
                    HISTORICAL
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <div className="research-status-pill">
          <span className="dot-indicator" />
          RESEARCH ANALYSIS — NOT AN OFFICIAL WARNING
        </div>
      </div>

      {/* Historical vs Recent Observation Callout Banner */}
      {selectedDate === '2024-07-28' && (
        <div style={{ margin: '16px 0', padding: '12px 16px', background: 'rgba(56, 189, 248, 0.08)', border: '1px solid rgba(56, 189, 248, 0.25)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <strong style={{ color: '#38bdf8', fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              RECENT SATELLITE OBSERVATION (28 Jul 2024)
            </strong>
            <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#94a3b8' }}>
              Satellite: Sentinel-1 C-SAR IW GRD | Resolution: 10 m | Status: <strong>MODEL-INFERRED FLOOD EXTENT</strong> (Passed distribution gate; no direct ground-truth validation available).
            </p>
          </div>
          <button
            type="button"
            className="text-btn"
            style={{ fontSize: '12px', color: '#38bdf8', background: 'transparent', border: '1px solid #38bdf8', borderRadius: 4, padding: '4px 10px', cursor: 'pointer' }}
            onClick={() => setSelectedDate('2022-10-15')}
          >
            ⇄ Compare with Historical 2022 Event
          </button>
        </div>
      )}
      {selectedDate === '2022-10-15' && (
        <div style={{ margin: '16px 0', padding: '12px 16px', background: 'rgba(16, 185, 129, 0.08)', border: '1px solid rgba(16, 185, 129, 0.25)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <strong style={{ color: '#10b981', fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              HISTORICAL ANALYSIS (15 Oct 2022 — Baseline Benchmark)
            </strong>
            <p style={{ margin: '4px 0 0', fontSize: '12px', color: '#94a3b8' }}>
              Satellite: Sentinel-1 SAR + Sentinel-2 L2A | Resolution: 10 m | Status: <strong>VALIDATED BENCHMARK</strong> (Includes BFCD-22 ground-truth damage splits in Muzaffarpur).
            </p>
          </div>
          <button
            type="button"
            className="text-btn"
            style={{ fontSize: '12px', color: '#10b981', background: 'transparent', border: '1px solid #10b981', borderRadius: 4, padding: '4px 10px', cursor: 'pointer' }}
            onClick={() => setSelectedDate('2024-07-28')}
          >
            ⇄ Inspect Recent 2024 Satellite Scene
          </button>
        </div>
      )}

      {/* Main Impact Overview Cards */}
      <div className="impact-summary-grid">
        <div className="impact-card highlight-cyan">
          <div className="impact-card-title">
            {eventData.date === '2024-07-28' ? 'Model-Inferred Flooded Area' : 'Observed Flooded Area'}
          </div>
          <div className="impact-card-value">
            {eventData.total_flooded_km2.toLocaleString()} <span className="unit">km²</span>
          </div>
          <div className="impact-card-sub">
            {eventData.flooded_pct}% of Bihar analyzed area (Sentinel-1 C-SAR 10m)
          </div>
        </div>

        <div className="impact-card highlight-emerald">
          <div className="impact-card-title">Affected Cropland</div>
          <div className="impact-card-value">
            {eventData.affected_cropland_km2.toLocaleString()} <span className="unit">km²</span>
          </div>
          <div className="impact-card-sub">Agricultural acreage intersecting inundation footprint</div>
        </div>

        <div className="impact-card highlight-amber">
          <div className="impact-card-title">Exposed Settlement Footprint</div>
          <div className="impact-card-value">
            {eventData.affected_buildings != null ? (
              <>~{eventData.affected_buildings.toLocaleString()} <span className="unit">structures</span></>
            ) : (
              <span className="unit" style={{ fontSize: '15px', color: 'var(--fg-dim)' }}>Unmeasured (null)</span>
            )}
          </div>
          <div className="impact-card-sub">
            {eventData.affected_buildings != null ? 'Census & WorldPop building density overlay' : 'Settlement layer unmeasured for this observation'}
          </div>
        </div>

        <div className="impact-card highlight-purple">
          <div className="impact-card-title">Affected Road Network</div>
          <div className="impact-card-value">
            {eventData.affected_roads_km != null ? (
              <>{eventData.affected_roads_km.toLocaleString()} <span className="unit">km</span></>
            ) : (
              <span className="unit" style={{ fontSize: '15px', color: 'var(--fg-dim)' }}>Unmeasured (null)</span>
            )}
          </div>
          <div className="impact-card-sub">
            {eventData.affected_roads_km != null ? 'Major highways and arterial routes intersecting water' : 'Road network layer unmeasured for this observation'}
          </div>
        </div>
      </div>

      {/* Muzaffarpur BFCD-22 Crop Damage Deep Dive */}
      <div className="crop-damage-banner">
        <div className="crop-banner-header">
          <div className="crop-banner-title">
            🌾 Cropland Damage Breakdown — {BIHAR_DATA.bfcd22_crop_damage.region} (BFCD-22 / Sentinel-2)
          </div>
          <span className="badge-crop">Model: CropDamageUNet (Dual-Temporal NDVI)</span>
        </div>
        <p className="crop-banner-desc">
          Evaluated on the October 2022 post-monsoon flood event using pre-flood and post-flood Sentinel-2 optical imagery.
          Categorized into 3 distinct damage classes:
        </p>

        <div className="crop-classes-grid">
          {BIHAR_DATA.bfcd22_crop_damage.classes.map((c) => (
            <div key={c.id} className="crop-class-card" style={{ borderTop: `3px solid ${c.color}` }}>
              <div className="crop-class-header">
                <span className="crop-class-name" style={{ color: c.color }}>
                  Class {c.id}: {c.name}
                </span>
                <span className="crop-class-pct">{c.pct}%</span>
              </div>
              <div className="crop-class-area">{c.area_km2} km²</div>
              <div className="crop-class-note">{c.description}</div>
            </div>
          ))}
        </div>
        <div className="crop-disclaimer">
          <strong>Validation Guard: </strong>
          Mean IoU: {BIHAR_DATA.bfcd22_crop_damage.mean_iou} | Macro F1: {BIHAR_DATA.bfcd22_crop_damage.macro_f1}.
          This is an experimental research model output. It does NOT represent an official statutory crop compensation survey.
        </div>
      </div>

      {/* Interactive Bihar Impact Map & District Explorer */}
      <div className="map-explorer-layout">
        <div className="map-panel">
          <div className="map-panel-header">
            <div className="map-title">Interactive Bihar Flood Impact Map</div>
            <div className="layer-toggles">
              <button
                type="button"
                className={`layer-chip ${activeLayers.floodExtent ? 'active' : ''}`}
                onClick={() => toggleLayer('floodExtent')}
              >
                🌊 Flood Extent (Observed)
              </button>
              <button
                type="button"
                className={`layer-chip ${activeLayers.historicalHazard ? 'active' : ''}`}
                onClick={() => toggleLayer('historicalHazard')}
              >
                📜 Historical Hazard (NRSC)
              </button>
              <button
                type="button"
                className={`layer-chip ${activeLayers.cropland ? 'active' : ''}`}
                onClick={() => toggleLayer('cropland')}
              >
                🌾 Cropland
              </button>
              <button
                type="button"
                className={`layer-chip ${activeLayers.buildings ? 'active' : ''}`}
                onClick={() => toggleLayer('buildings')}
              >
                🏘️ Buildings
              </button>
              <button
                type="button"
                className={`layer-chip ${activeLayers.roads ? 'active' : ''}`}
                onClick={() => toggleLayer('roads')}
              >
                🛣️ Roads
              </button>
            </div>
          </div>

          {/* SVG Map Canvas representation */}
          <div className="map-canvas-container">
            <svg
              className="bihar-svg-map"
              viewBox="0 0 800 480"
              xmlns="http://www.w3.org/2000/svg"
            >
              {/* Background Map Grid */}
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="1" />
                </pattern>
                <radialGradient id="waterGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
                </radialGradient>
              </defs>

              <rect width="800" height="480" fill="#0d1117" />
              <rect width="800" height="480" fill="url(#grid)" />

              {/* Ganga River Flow representation */}
              <path
                d="M 60 270 Q 200 280 340 260 T 520 270 T 740 310"
                fill="none"
                stroke="#1e3a8a"
                strokeWidth="8"
                strokeLinecap="round"
                opacity="0.6"
              />
              <text x="360" y="278" fill="#3b82f6" fontSize="11" opacity="0.8" fontWeight="600">
                Ganga River Basin
              </text>

              {/* North Bihar River Basins (Burhi Gandak & Kosi) */}
              <path
                d="M 220 70 Q 250 150 310 260"
                fill="none"
                stroke="#0284c7"
                strokeWidth="4"
                strokeDasharray="4 2"
                opacity="0.4"
              />
              <path
                d="M 540 60 Q 520 160 510 270"
                fill="none"
                stroke="#0284c7"
                strokeWidth="5"
                opacity="0.4"
              />
              <text x="525" y="140" fill="#38bdf8" fontSize="10" opacity="0.6">
                Kosi Basin
              </text>

              {/* District Polygons */}
              {/* Muzaffarpur */}
              <g
                className={`district-node ${selectedDistrict === 'Muzaffarpur' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Muzaffarpur')}
              >
                <polygon
                  points="260,170 330,160 350,210 290,230 250,200"
                  fill={
                    selectedDistrict === 'Muzaffarpur'
                      ? 'rgba(6, 182, 212, 0.45)'
                      : activeLayers.floodExtent
                      ? 'rgba(6, 182, 212, 0.25)'
                      : 'rgba(255,255,255,0.06)'
                  }
                  stroke={selectedDistrict === 'Muzaffarpur' ? '#06b6d4' : '#334155'}
                  strokeWidth={selectedDistrict === 'Muzaffarpur' ? '2.5' : '1'}
                />
                <circle cx="300" cy="195" r="5" fill="#06b6d4" />
                <text x="270" y="215" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Muzaffarpur
                </text>
              </g>

              {/* Darbhanga */}
              <g
                className={`district-node ${selectedDistrict === 'Darbhanga' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Darbhanga')}
              >
                <polygon
                  points="335,140 405,130 420,195 355,205"
                  fill={
                    selectedDistrict === 'Darbhanga'
                      ? 'rgba(6, 182, 212, 0.45)'
                      : activeLayers.floodExtent
                      ? 'rgba(6, 182, 212, 0.22)'
                      : 'rgba(255,255,255,0.06)'
                  }
                  stroke={selectedDistrict === 'Darbhanga' ? '#06b6d4' : '#334155'}
                  strokeWidth={selectedDistrict === 'Darbhanga' ? '2.5' : '1'}
                />
                <circle cx="375" cy="165" r="4.5" fill="#06b6d4" />
                <text x="350" y="180" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Darbhanga
                </text>
              </g>

              {/* Samastipur */}
              <g
                className={`district-node ${selectedDistrict === 'Samastipur' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Samastipur')}
              >
                <polygon
                  points="355,210 420,200 440,250 365,260"
                  fill={
                    selectedDistrict === 'Samastipur'
                      ? 'rgba(6, 182, 212, 0.45)'
                      : activeLayers.floodExtent
                      ? 'rgba(6, 182, 212, 0.20)'
                      : 'rgba(255,255,255,0.06)'
                  }
                  stroke={selectedDistrict === 'Samastipur' ? '#06b6d4' : '#334155'}
                  strokeWidth={selectedDistrict === 'Samastipur' ? '2.5' : '1'}
                />
                <text x="370" y="240" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Samastipur
                </text>
              </g>

              {/* Supaul */}
              <g
                className={`district-node ${selectedDistrict === 'Supaul' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Supaul')}
              >
                <polygon
                  points="470,80 540,85 530,165 460,155"
                  fill={
                    selectedDistrict === 'Supaul'
                      ? 'rgba(6, 182, 212, 0.45)'
                      : activeLayers.floodExtent
                      ? 'rgba(6, 182, 212, 0.20)'
                      : 'rgba(255,255,255,0.06)'
                  }
                  stroke={selectedDistrict === 'Supaul' ? '#06b6d4' : '#334155'}
                  strokeWidth={selectedDistrict === 'Supaul' ? '2.5' : '1'}
                />
                <text x="480" y="130" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Supaul
                </text>
              </g>

              {/* Saharsa */}
              <g
                className={`district-node ${selectedDistrict === 'Saharsa' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Saharsa')}
              >
                <polygon
                  points="465,160 525,170 515,225 450,215"
                  fill={
                    selectedDistrict === 'Saharsa'
                      ? 'rgba(6, 182, 212, 0.45)'
                      : activeLayers.floodExtent
                      ? 'rgba(6, 182, 212, 0.18)'
                      : 'rgba(255,255,255,0.06)'
                  }
                  stroke={selectedDistrict === 'Saharsa' ? '#06b6d4' : '#334155'}
                  strokeWidth={selectedDistrict === 'Saharsa' ? '2.5' : '1'}
                />
                <text x="470" y="200" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Saharsa
                </text>
              </g>

              {/* Khagaria */}
              <g
                className={`district-node ${selectedDistrict === 'Khagaria' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Khagaria')}
              >
                <polygon
                  points="445,225 520,230 510,275 435,270"
                  fill={
                    selectedDistrict === 'Khagaria'
                      ? 'rgba(6, 182, 212, 0.45)'
                      : activeLayers.floodExtent
                      ? 'rgba(6, 182, 212, 0.22)'
                      : 'rgba(255,255,255,0.06)'
                  }
                  stroke={selectedDistrict === 'Khagaria' ? '#06b6d4' : '#334155'}
                  strokeWidth={selectedDistrict === 'Khagaria' ? '2.5' : '1'}
                />
                <text x="455" y="255" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Khagaria
                </text>
              </g>

              {/* Madhubani */}
              <g
                className={`district-node ${selectedDistrict === 'Madhubani' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Madhubani')}
              >
                <polygon
                  points="340,75 460,80 450,135 345,135"
                  fill={
                    selectedDistrict === 'Madhubani'
                      ? 'rgba(6, 182, 212, 0.45)'
                      : 'rgba(255,255,255,0.06)'
                  }
                  stroke={selectedDistrict === 'Madhubani' ? '#06b6d4' : '#334155'}
                  strokeWidth={selectedDistrict === 'Madhubani' ? '2.5' : '1'}
                />
                <text x="375" y="110" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Madhubani
                </text>
              </g>

              {/* Patna */}
              <g
                className={`district-node ${selectedDistrict === 'Patna' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Patna')}
              >
                <polygon
                  points="250,265 345,260 360,310 240,315"
                  fill={selectedDistrict === 'Patna' ? 'rgba(6, 182, 212, 0.45)' : 'rgba(255,255,255,0.04)'}
                  stroke={selectedDistrict === 'Patna' ? '#06b6d4' : '#334155'}
                  strokeWidth="1"
                />
                <text x="285" y="295" fill="#cbd5e1" fontSize="11">
                  Patna
                </text>
              </g>

              {/* Katihar */}
              <g
                className={`district-node ${selectedDistrict === 'Katihar' ? 'selected' : ''}`}
                onClick={() => setSelectedDistrict('Katihar')}
              >
                <polygon
                  points="600,190 680,200 660,285 580,270"
                  fill={selectedDistrict === 'Katihar' ? 'rgba(6, 182, 212, 0.45)' : 'rgba(6, 182, 212, 0.15)'}
                  stroke={selectedDistrict === 'Katihar' ? '#06b6d4' : '#334155'}
                  strokeWidth="1"
                />
                <text x="610" y="240" fill="#f8fafc" fontSize="11" fontWeight="600">
                  Katihar
                </text>
              </g>
            </svg>

            {/* Map Legend */}
            <div className="map-legend-card">
              <div className="legend-title">Map Layer Classification</div>
              <div className="legend-item">
                <span className="legend-indicator blue-box" />
                <span>
                  <strong>SATELLITE OBSERVATION:</strong> Sentinel-1 C-SAR ({selectedDate})
                </span>
              </div>
              <div className="legend-item">
                <span className="legend-indicator cyan-box" />
                <span>
                  <strong>MODEL-INFERRED FLOOD EXTENT:</strong> U-Net Radar Water Inference (10m)
                </span>
              </div>
              <div className="legend-item">
                <span className="legend-indicator orange-box" />
                <span>
                  <strong>REFERENCE LAYER:</strong> NRSC Atlas (1998–2019) Hazard Zonation &amp; Cropland
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* District Detail Card */}
        <div className="district-detail-card">
          <div className="detail-header">
            <div>
              <span className="detail-tag">DISTRICT PROFILE</span>
              <h3 className="detail-district-name">{activeDistrictData.name}</h3>
              <span className="detail-basin">Basin: {activeDistrictData.basin}</span>
            </div>
            <div className="hazard-badge-box">
              <span className={`hazard-badge ${activeDistrictData.hazard_class.toLowerCase().replace(' ', '-')}`}>
                {activeDistrictData.hazard_class} Hazard
              </span>
              <span className="hazard-sub">NRSC 1998–2019</span>
            </div>
          </div>

          <div className="detail-stats-grid">
            <div className="detail-stat-box">
              <div className="detail-stat-val">
                {activeDistrictData.flooded_km2} <span className="u">km²</span>
              </div>
              <div className="detail-stat-lbl">Inundated Area</div>
            </div>

            <div className="detail-stat-box">
              <div className="detail-stat-val">{activeDistrictData.flood_pct}%</div>
              <div className="detail-stat-lbl">Flooded Fraction</div>
            </div>

            <div className="detail-stat-box">
              <div className="detail-stat-val">
                {activeDistrictData.cropland_km2} <span className="u">km²</span>
              </div>
              <div className="detail-stat-lbl">Cropland Affected</div>
            </div>

            <div className="detail-stat-box">
              <div className="detail-stat-val">
                {activeDistrictData.impact_index} <span className="u">/ 1.0</span>
              </div>
              <div className="detail-stat-lbl">SAT-AI Impact Index</div>
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-title">Critical Exposure Figures</div>
            <div className="exposure-row">
              <span>🏠 Settlement / Building Structures:</span>
              <strong>{activeDistrictData.buildings.toLocaleString()}</strong>
            </div>
            <div className="exposure-row">
              <span>🛣️ Road Alignments Intersected:</span>
              <strong>{activeDistrictData.roads_km} km</strong>
            </div>
            <div className="exposure-row">
              <span>🌐 Administrative Area:</span>
              <strong>{activeDistrictData.total_area_km2.toLocaleString()} km²</strong>
            </div>
          </div>

          <div className="detail-section">
            <div className="detail-section-title">Inundated Blocks Inside Footprint</div>
            <div className="blocks-chips">
              {activeDistrictData.blocks.map((b) => (
                <span key={b} className="block-chip">
                  📍 {b}
                </span>
              ))}
            </div>
          </div>

          <div className="detail-footer-note">
            Acquisition Date: {selectedDate} • Projection: UTM Zone 45N (EPSG:32645)
          </div>
        </div>
      </div>

      {/* District Impact Table with Interactive Sorting */}
      <div className="impact-table-card">
        <div className="table-header-row">
          <div>
            <h3 className="table-title">District-Level Flood Impact Aggregation</h3>
            <p className="table-sub">
              Deterministic GIS overlay across 16 affected Bihar districts for {selectedDate}.
            </p>
          </div>

          <div className="sort-controls">
            <span className="sort-label">Sort By:</span>
            <button
              type="button"
              className={`sort-btn ${sortKey === 'flooded_km2' ? 'active' : ''}`}
              onClick={() => setSortKey('flooded_km2')}
            >
              Flooded Area
            </button>
            <button
              type="button"
              className={`sort-btn ${sortKey === 'flood_pct' ? 'active' : ''}`}
              onClick={() => setSortKey('flood_pct')}
            >
              Flood %
            </button>
            <button
              type="button"
              className={`sort-btn ${sortKey === 'cropland_km2' ? 'active' : ''}`}
              onClick={() => setSortKey('cropland_km2')}
            >
              Cropland
            </button>
            <button
              type="button"
              className={`sort-btn ${sortKey === 'impact_index' ? 'active' : ''}`}
              onClick={() => setSortKey('impact_index')}
            >
              Impact Index
            </button>
          </div>
        </div>

        <div className="table-responsive">
          <table className="bihar-impact-table">
            <thead>
              <tr>
                <th>District</th>
                <th>Flooded Area</th>
                <th>Flood %</th>
                <th>Cropland Affected</th>
                <th>Roads (km)</th>
                <th>Buildings</th>
                <th>Historical Hazard</th>
                <th>SAT-AI Impact Index</th>
              </tr>
            </thead>
            <tbody>
              {sortedDistricts.map((d) => (
                <tr
                  key={d.name}
                  className={selectedDistrict === d.name ? 'highlight-row' : ''}
                  onClick={() => setSelectedDistrict(d.name)}
                >
                  <td className="district-name-cell">
                    <strong>{d.name}</strong>
                  </td>
                  <td>
                    <span className="mono-num">{d.flooded_km2} km²</span>
                  </td>
                  <td>
                    <span className="mono-num">{d.flood_pct}%</span>
                  </td>
                  <td>
                    <span className="mono-num">{d.cropland_km2} km²</span>
                  </td>
                  <td>
                    <span className="mono-num">{d.roads_km} km</span>
                  </td>
                  <td>
                    <span className="mono-num">{d.buildings.toLocaleString()}</span>
                  </td>
                  <td>
                    <span className={`table-hazard-tag ${d.hazard_class.toLowerCase().replace(' ', '-')}`}>
                      {d.hazard_class}
                    </span>
                  </td>
                  <td>
                    <span className="impact-index-pill">{d.impact_index}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Collapsible Scientific Evidence Drawer */}
        <div className="evidence-accordion-wrapper">
          <button
            type="button"
            className="toggle-evidence-btn"
            onClick={() => setEvidenceOpen((prev) => !prev)}
          >
            <span>{evidenceOpen ? '▼ Hide Evidence & Scientific Provenance' : '▶ Show Evidence & Scientific Provenance'}</span>
          </button>

          {evidenceOpen && (
            <div className="evidence-content-panel">
              <div className="evidence-grid">
                <div className="evidence-box">
                  <div className="evidence-box-title">Area Calculation Methodology</div>
                  <p>
                    Area figures are computed via UTM Zone 45N (EPSG:32645), a projected CRS used for metric area calculations.
                    Naive pixel counting across geographic degrees is prohibited to avoid latitudinal cosine distortion.
                  </p>
                </div>
                <div className="evidence-box">
                  <div className="evidence-box-title">Historical Flood Hazard Data</div>
                  <p>
                    Hazard ratings are sourced directly from the NRSC/ISRO Bihar Flood Hazard Zonation Atlas (1998–2019),
                    categorizing multi-decadal flood frequency over 22 years. This is strictly separated from current inundation.
                  </p>
                </div>
                <div className="evidence-box">
                  <div className="evidence-box-title">Crop Damage Model (BFCD-22)</div>
                  <p>
                    Crop damage segmentation operates on pre/post Sentinel-2 optical bands and Delta-NDVI, trained on the
                    Muzaffarpur October 2022 flood benchmark split (Macro F1 0.742, Mean IoU 0.6463).
                  </p>
                </div>
                <div className="evidence-box">
                  <div className="evidence-box-title">SAT-AI Impact Index Weights</div>
                  <p>
                    Impact Index = 0.40 × Flood Fraction + 0.25 × Cropland Exposure + 0.20 × Building Exposure + 0.15 × Road Exposure.
                    This is a transparent multi-criteria research metric, NOT an official government severity rating or economic damage estimate.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
