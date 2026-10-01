'use client';

import { useState } from 'react';
import {
  PLACES_DATABASE,
  SEARCH_INDEX,
  getPlaceProfile,
  type PlaceProfile,
  type HistoricalEvent,
} from '@/lib/places-data';
import BIHAR_DATA from '@/lib/bihar-impact-data.json';

interface PlaceProfileProps {
  initialPlaceId?: string;
  onSelectPlaceForChat?: (placeName: string) => void;
}

export function PlaceProfileSection({
  initialPlaceId = 'bihar',
  onSelectPlaceForChat,
}: PlaceProfileProps) {
  const [selectedPlaceId, setSelectedPlaceId] = useState<string>(initialPlaceId);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<typeof SEARCH_INDEX>([]);
  const [selectedEventIndex, setSelectedEventIndex] = useState<number>(4); // Default to 2022 event
  const [activeDistrict, setActiveDistrict] = useState<string>('Muzaffarpur');
  const [activeLayers, setActiveLayers] = useState({
    floodExtent: true,
    cropland: true,
    waterBodies: true,
    settlements: false,
    roads: false,
    historicalHazard: true,
  });
  const [evidenceOpen, setEvidenceOpen] = useState(false);

  const place: PlaceProfile = getPlaceProfile(selectedPlaceId);
  const currentEvent: HistoricalEvent =
    place.timeline[selectedEventIndex] || place.timeline[place.timeline.length - 1] || place.timeline[0];

  const handleSearchChange = (val: string) => {
    setSearchQuery(val);
    if (!val.trim()) {
      setSearchResults([]);
      return;
    }
    const q = val.toLowerCase().trim();
    const matches = SEARCH_INDEX.filter(
      (item) => item.name.toLowerCase().includes(q) || item.parent.toLowerCase().includes(q)
    ).slice(0, 6);
    setSearchResults(matches);
  };

  const handleSelectPlace = (id: string, name: string) => {
    setSelectedPlaceId(id);
    setSearchQuery('');
    setSearchResults([]);
    setSelectedEventIndex(0);
    if (onSelectPlaceForChat) {
      onSelectPlaceForChat(name);
    }
  };

  const toggleLayer = (layer: keyof typeof activeLayers) => {
    setActiveLayers((prev) => ({ ...prev, [layer]: !prev[layer] }));
  };

  const districtData =
    BIHAR_DATA.districts.find((d) => d.name.toLowerCase() === activeDistrict.toLowerCase()) ||
    BIHAR_DATA.districts[0];

  return (
    <section id="explore" className="place-profile-section section-padding">
      <div className="section-header centered">
        <div className="section-tag">Geospatial Intelligence Platform</div>
        <h2 className="section-title">
          Where Do You Want to <span className="gradient-text">Look?</span>
        </h2>
        <p className="section-lead">
          Explore environmental conditions, disaster history, satellite observations, and AI-powered spatial insights.
        </p>

        {/* Global Place Search Input */}
        <div className="place-search-container">
          <div className="place-search-bar">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              className="place-search-input"
              placeholder="Search a place, district or region (e.g. Bihar, Muzaffarpur, Darbhanga, Mumbai, Assam)..."
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="clear-search-btn"
                onClick={() => {
                  setSearchQuery('');
                  setSearchResults([]);
                }}
              >
                ✕
              </button>
            )}
          </div>

          {/* Autocomplete Dropdown */}
          {searchResults.length > 0 && (
            <div className="place-search-dropdown">
              {searchResults.map((item) => (
                <button
                  key={`${item.id}-${item.name}`}
                  type="button"
                  className="dropdown-item"
                  onClick={() => handleSelectPlace(item.id, item.name)}
                >
                  <span className="item-name">{item.name}</span>
                  <span className="item-meta">
                    {item.type} • {item.parent}
                  </span>
                  {item.isAvailable && <span className="item-badge">Full Analysis</span>}
                </button>
              ))}
            </div>
          )}

          {/* Quick Filter Place Chips */}
          <div className="quick-place-chips">
            <span className="chips-label">Quick Places:</span>
            {[
              { id: 'bihar', label: 'Bihar (Showcase)' },
              { id: 'muzaffarpur', label: 'Muzaffarpur' },
              { id: 'darbhanga', label: 'Darbhanga' },
              { id: 'mumbai', label: 'Mumbai' },
              { id: 'assam', label: 'Assam' },
            ].map((p) => (
              <button
                key={p.id}
                type="button"
                className={`quick-chip ${selectedPlaceId === p.id ? 'active' : ''}`}
                onClick={() => handleSelectPlace(p.id, p.label.split(' ')[0])}
              >
                📍 {p.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ====================================================================
          1. PLACE PROFILE HEADER
          ==================================================================== */}
      <div className="place-header-banner">
        <div className="place-title-group">
          <div className="place-hierarchy">
            <span>{place.country}</span>
            <span className="sep">/</span>
            <span>{place.state}</span>
            {place.adminLevel === 'District' && (
              <>
                <span className="sep">/</span>
                <span className="highlight-tag">{place.name} District</span>
              </>
            )}
          </div>
          <h1 className="place-main-name">{place.name}</h1>
          <div className="place-tagline">{place.tagline}</div>
        </div>

        <div className="place-meta-stats">
          <div className="stat-pill">
            <span className="stat-label">Total Area</span>
            <span className="stat-val">{place.totalAreaKm2.toLocaleString()} km²</span>
          </div>
          <div className="stat-pill">
            <span className="stat-label">Coordinates</span>
            <span className="stat-val mono">
              {place.coordinates[0].toFixed(2)}°N, {place.coordinates[1].toFixed(2)}°E
            </span>
          </div>
          <div className="stat-pill highlight-cyan">
            <span className="stat-label">Analysis State</span>
            <span className="stat-val">
              {place.hasFullAnalysis ? '✓ Active Satellite Layers' : 'Archived Observation'}
            </span>
          </div>
        </div>
      </div>

      <div className="place-overview-text">
        <p>{place.overview}</p>
      </div>

      {/* ====================================================================
          2. ENVIRONMENTAL PROFILE (5 CARDS)
          ==================================================================== */}
      <div className="profile-sub-section">
        <div className="sub-section-header">
          <span className="sub-icon">🌿</span>
          <div>
            <h3 className="sub-title">Environmental Landscape</h3>
            <p className="sub-desc">
              Authoritative land-cover, hydrological, and terrain characteristics derived from satellite data and survey records.
            </p>
          </div>
        </div>

        <div className="environmental-cards-grid">
          {/* Card 1: Vegetation */}
          <div className="env-card">
            <div className="env-card-header">
              <span className="env-icon">🌿</span>
              <span className="env-card-title">{place.environment.vegetation.title}</span>
            </div>
            <div className="env-card-metric">
              {place.environment.vegetation.forestCoverKm2 ? (
                <>
                  <span className="metric-num">
                    {place.environment.vegetation.forestCoverKm2.toLocaleString()}
                  </span>
                  <span className="metric-unit">km²</span>
                  <span className="metric-sub">
                    ({place.environment.vegetation.forestCoverPct}% of land)
                  </span>
                </>
              ) : (
                <span className="metric-num-na">Urban / Non-forest</span>
              )}
            </div>
            <ul className="env-list">
              {place.environment.vegetation.dominantTypes.map((t, idx) => (
                <li key={idx}>• {t}</li>
              ))}
            </ul>
            <div className="env-dynamics">
              <strong>NDVI Range: </strong>
              <span className="mono">{place.environment.vegetation.ndviRange}</span>
              <p>{place.environment.vegetation.seasonalDynamics}</p>
            </div>
            <div className="env-source">Source: {place.environment.vegetation.source}</div>
          </div>

          {/* Card 2: Agriculture */}
          <div className="env-card">
            <div className="env-card-header">
              <span className="env-icon">🌾</span>
              <span className="env-card-title">{place.environment.agriculture.title}</span>
            </div>
            <div className="env-card-metric">
              {place.environment.agriculture.netCroppedAreaKm2 ? (
                <>
                  <span className="metric-num">
                    {place.environment.agriculture.netCroppedAreaKm2.toLocaleString()}
                  </span>
                  <span className="metric-unit">km²</span>
                  <span className="metric-sub">
                    ({place.environment.agriculture.croppedAreaPct}% of land)
                  </span>
                </>
              ) : (
                <span className="metric-num-na">Minimal Agriculture</span>
              )}
            </div>
            <div className="env-crops-tags">
              {place.environment.agriculture.majorCrops.map((c, idx) => (
                <span key={idx} className="crop-tag">
                  {c}
                </span>
              ))}
            </div>
            <div className="env-dynamics">
              <strong>Cropping Intensity: </strong>
              <span>{place.environment.agriculture.croppingIntensity}</span>
              <p>{place.environment.agriculture.floodplainFarming}</p>
            </div>
            <div className="env-source">Source: {place.environment.agriculture.source}</div>
          </div>

          {/* Card 3: Water */}
          <div className="env-card">
            <div className="env-card-header">
              <span className="env-icon">💧</span>
              <span className="env-card-title">{place.environment.water.title}</span>
            </div>
            <div className="env-card-metric">
              {place.environment.water.surfaceWaterAreaKm2 ? (
                <>
                  <span className="metric-num">
                    {place.environment.water.surfaceWaterAreaKm2.toLocaleString()}
                  </span>
                  <span className="metric-unit">km²</span>
                  <span className="metric-sub">water spread</span>
                </>
              ) : (
                <span className="metric-num-na">Coastal Waters</span>
              )}
            </div>
            <div className="env-river-list">
              <strong>Major Rivers:</strong>
              <p>{place.environment.water.majorRivers.join(' • ')}</p>
            </div>
            <div className="env-dynamics">
              <p>{place.environment.water.basinDescription}</p>
              <small>{place.environment.water.seasonalWaterBodies}</small>
            </div>
            <div className="env-source">Source: {place.environment.water.source}</div>
          </div>

          {/* Card 4: Terrain */}
          <div className="env-card">
            <div className="env-card-header">
              <span className="env-icon">🏔</span>
              <span className="env-card-title">{place.environment.terrain.title}</span>
            </div>
            <div className="env-card-metric">
              <span className="metric-num">{place.environment.terrain.elevationRangeM}</span>
            </div>
            <div className="env-dynamics">
              <strong>Slope: </strong>
              <span>{place.environment.terrain.averageSlope}</span>
              <p>{place.environment.terrain.geomorphology}</p>
              <small>Drainage: {place.environment.terrain.drainagePattern}</small>
            </div>
            <div className="env-source">Source: {place.environment.terrain.source}</div>
          </div>

          {/* Card 5: Built Environment */}
          <div className="env-card">
            <div className="env-card-header">
              <span className="env-icon">🏙</span>
              <span className="env-card-title">{place.environment.builtEnvironment.title}</span>
            </div>
            <div className="env-card-metric">
              <span className="metric-num">
                {place.environment.builtEnvironment.arterialRoadsKm
                  ? `${place.environment.builtEnvironment.arterialRoadsKm.toLocaleString()} km`
                  : 'Urban Grid'}
              </span>
              <span className="metric-sub">arterial network</span>
            </div>
            <div className="env-dynamics">
              <strong>Settlement Pattern: </strong>
              <p>{place.environment.builtEnvironment.settlementCount}</p>
              <small>Density: {place.environment.builtEnvironment.settlementDensity}</small>
            </div>
            <div className="env-urban-centers">
              {place.environment.builtEnvironment.urbanCenters.map((u, idx) => (
                <span key={idx} className="urban-tag">
                  {u}
                </span>
              ))}
            </div>
            <div className="env-source">Source: {place.environment.builtEnvironment.source}</div>
          </div>
        </div>
      </div>

      {/* ====================================================================
          3. INTERACTIVE LANDSCAPE MAP
          ==================================================================== */}
      <div id="landscape-map" className="profile-sub-section">
        <div className="sub-section-header">
          <span className="sub-icon">🗺️</span>
          <div>
            <h3 className="sub-title">Landscape &amp; Multi-Layer Map</h3>
            <p className="sub-desc">
              Explore dynamic satellite inundation, agricultural land cover, infrastructure exposure, and 22-year historical flood frequency.
            </p>
          </div>
        </div>

        <div className="landscape-map-wrapper">
          {/* Map Layer Controls Bar */}
          <div className="map-toolbar">
            <div className="layer-toggles-group">
              <span className="toolbar-label">Active Layers:</span>
              <button
                type="button"
                className={`layer-btn ${activeLayers.floodExtent ? 'active' : ''}`}
                onClick={() => toggleLayer('floodExtent')}
              >
                <span className="dot cyan" /> Flood Extent (Observed)
              </button>
              <button
                type="button"
                className={`layer-btn ${activeLayers.cropland ? 'active' : ''}`}
                onClick={() => toggleLayer('cropland')}
              >
                <span className="dot emerald" /> Cropland (Agricultural)
              </button>
              <button
                type="button"
                className={`layer-btn ${activeLayers.waterBodies ? 'active' : ''}`}
                onClick={() => toggleLayer('waterBodies')}
              >
                <span className="dot blue" /> River Basin (Hydrology)
              </button>
              <button
                type="button"
                className={`layer-btn ${activeLayers.settlements ? 'active' : ''}`}
                onClick={() => toggleLayer('settlements')}
              >
                <span className="dot purple" /> Settlements (Census 2011)
              </button>
              <button
                type="button"
                className={`layer-btn ${activeLayers.roads ? 'active' : ''}`}
                onClick={() => toggleLayer('roads')}
              >
                <span className="dot orange" /> Roads (MoRTH)
              </button>
              <button
                type="button"
                className={`layer-btn ${activeLayers.historicalHazard ? 'active' : ''}`}
                onClick={() => toggleLayer('historicalHazard')}
              >
                <span className="dot amber" /> Hazard Zonation (NRSC)
              </button>
            </div>

            <div className="map-legend-pills">
              <span className="legend-tag">
                <i style={{ background: '#38bdf8' }} /> SATELLITE OBSERVATION
              </span>
              <span className="legend-tag">
                <i style={{ background: '#10b981' }} /> MODEL-INFERRED FLOOD EXTENT
              </span>
              <span className="legend-tag">
                <i style={{ background: '#f59e0b' }} /> REFERENCE LAYER
              </span>
            </div>
          </div>

          {/* Interactive Bihar/Regional SVG Geospatial Canvas */}
          <div className="map-canvas-container">
            <svg
              className="interactive-svg-canvas"
              viewBox="0 0 900 520"
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                <pattern id="river-mesh" width="20" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 0 10 Q 5 0, 10 10 T 20 10" fill="none" stroke="#0284c7" strokeWidth="0.8" opacity="0.3" />
                </pattern>
                <pattern id="crop-pattern" width="12" height="12" patternUnits="userSpaceOnUse">
                  <circle cx="6" cy="6" r="1" fill="#10b981" opacity="0.4" />
                </pattern>
              </defs>

              {/* State boundary background */}
              <rect x="0" y="0" width="900" height="520" fill="#030712" />

              {/* Major Ganga & Tributary River Paths */}
              {activeLayers.waterBodies && (
                <g className="rivers-layer">
                  {/* Ganga Mainstem */}
                  <path
                    d="M 50 310 Q 250 320, 420 300 T 650 310 T 850 330"
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth="7"
                    opacity="0.6"
                  />
                  {/* Gandak */}
                  <path
                    d="M 160 50 Q 240 180, 410 300"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="3.5"
                    opacity="0.5"
                  />
                  {/* Burhi Gandak */}
                  <path
                    d="M 270 50 Q 340 160, 520 305"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="3.5"
                    opacity="0.5"
                  />
                  {/* Bagmati / Kamla */}
                  <path
                    d="M 370 40 Q 420 140, 560 305"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="3"
                    opacity="0.5"
                  />
                  {/* Kosi River */}
                  <path
                    d="M 580 40 Q 610 160, 680 315"
                    fill="none"
                    stroke="#38bdf8"
                    strokeWidth="5"
                    opacity="0.6"
                  />
                  {/* Son River from south */}
                  <path
                    d="M 320 480 Q 360 380, 390 310"
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth="4"
                    opacity="0.5"
                  />
                </g>
              )}

              {/* District Polygons */}
              <g className="districts-polygons">
                {BIHAR_DATA.districts.map((d, idx) => {
                  const isSelected = activeDistrict.toLowerCase() === d.name.toLowerCase();
                  // Pre-computed spatial layout positions for Bihar map representation
                  const col = idx % 7;
                  const row = Math.floor(idx / 7);
                  const x = 75 + col * 110 + (row % 2 === 1 ? 25 : 0);
                  const y = 80 + row * 90;

                  // Fill color based on active layers
                  let fillColor = '#0f172a';
                  if (activeLayers.floodExtent && d.flood_pct > 12) {
                    fillColor = 'rgba(56, 189, 248, 0.28)';
                  } else if (activeLayers.cropland) {
                    fillColor = 'rgba(16, 185, 129, 0.12)';
                  } else if (activeLayers.historicalHazard && d.hazard_class === 'Very High') {
                    fillColor = 'rgba(244, 63, 94, 0.2)';
                  }

                  return (
                    <g
                      key={d.name}
                      className="district-map-node"
                      onClick={() => setActiveDistrict(d.name)}
                      style={{ cursor: 'pointer' }}
                    >
                      <polygon
                        points={`${x - 48},${y - 35} ${x + 48},${y - 35} ${x + 55},${y + 35} ${x - 40},${y + 40}`}
                        fill={fillColor}
                        stroke={isSelected ? '#38bdf8' : '#334155'}
                        strokeWidth={isSelected ? 2.5 : 1}
                        className="district-shape"
                      />
                      <circle
                        cx={x}
                        cy={y}
                        r={isSelected ? 5 : 3}
                        fill={isSelected ? '#38bdf8' : '#94a3b8'}
                      />
                      <text x={x} y={y + 14} className="district-svg-text">
                        {d.name}
                      </text>
                      {activeLayers.floodExtent && d.flood_pct > 5 && (
                        <text x={x} y={y - 8} className="district-svg-pct">
                          {d.flood_pct}%
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>

              {/* Road Network Overlay */}
              {activeLayers.roads && (
                <g className="roads-layer">
                  <path d="M 60 280 L 840 290" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="6,3" opacity="0.6" />
                  <path d="M 430 60 L 410 460" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="6,3" opacity="0.6" />
                  <path d="M 280 60 L 680 460" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4,4" opacity="0.4" />
                </g>
              )}
            </svg>

            {/* District Inspector Card Overlay */}
            <div className="map-district-inspector">
              <div className="inspector-header">
                <span className="inspector-badge">Selected District</span>
                <h4 className="inspector-name">{districtData.name}</h4>
                <span className="inspector-basin">Basin: {districtData.basin}</span>
              </div>

              <div className="inspector-metrics">
                <div className="insp-row">
                  <span>Flooded Extent:</span>
                  <strong className="cyan">{districtData.flooded_km2} km² ({districtData.flood_pct}%)</strong>
                </div>
                <div className="insp-row">
                  <span>Cropland Affected:</span>
                  <strong className="emerald">{districtData.cropland_km2} km²</strong>
                </div>
                <div className="insp-row">
                  <span>Building Footprints:</span>
                  <strong>{districtData.buildings.toLocaleString()} structures</strong>
                </div>
                <div className="insp-row">
                  <span>Arterial Roads:</span>
                  <strong>{districtData.roads_km} km</strong>
                </div>
                <div className="insp-row">
                  <span>Historical Hazard:</span>
                  <span className={`hazard-chip ${districtData.hazard_class.toLowerCase().replace(' ', '-')}`}>
                    {districtData.hazard_class}
                  </span>
                </div>
                <div className="insp-row highlight-box">
                  <span>SAT-AI Impact Index:</span>
                  <strong className="cyan">{districtData.impact_index}</strong>
                </div>
              </div>

              <div className="inspector-blocks">
                <small>Most Affected Blocks:</small>
                <div className="blocks-list">
                  {districtData.blocks.map((b, idx) => (
                    <span key={idx} className="block-pill">
                      {b}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ====================================================================
          4. DISASTER HISTORY & HAZARD PROFILE
          ==================================================================== */}
      <div id="disasters" className="profile-sub-section">
        <div className="sub-section-header">
          <span className="sub-icon">⚡</span>
          <div>
            <h3 className="sub-title">Disaster Profile &amp; Historical Hazard</h3>
            <p className="sub-desc">
              Multi-decadal frequency classification across major natural hazard categories based on authoritative registries.
            </p>
          </div>
        </div>

        <div className="hazards-grid">
          {place.hazards.map((h) => {
            const getIcon = (type: string) => {
              switch (type) {
                case 'Flood':
                  return '🌊';
                case 'Extreme Rainfall':
                  return '🌧';
                case 'Earthquake':
                  return '🌍';
                case 'Wildfire':
                  return '🔥';
                case 'Cyclone':
                  return '🌪';
                default:
                  return '⚠️';
              }
            };

            return (
              <div key={h.hazard} className="hazard-card">
                <div className="hazard-card-top">
                  <span className="hazard-icon">{getIcon(h.hazard)}</span>
                  <span className="hazard-name">{h.hazard}</span>
                  <span className={`hazard-rating-tag ${h.rating.toLowerCase().replace(' ', '-')}`}>
                    {h.rating}
                  </span>
                </div>
                <div className="hazard-frequency">{h.frequency}</div>
                <p className="hazard-context">{h.historicalContext}</p>
                <div className="hazard-source">Source: {h.source}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ====================================================================
          5. DISASTER TIMELINE & EVENTS
          ==================================================================== */}
      <div id="timeline" className="profile-sub-section">
        <div className="sub-section-header">
          <span className="sub-icon">📅</span>
          <div>
            <h3 className="sub-title">Disaster Timeline &amp; Satellite Observations</h3>
            <p className="sub-desc">
              Select an archived flood event to inspect dual-temporal satellite captures, observed water spread, and cropland impact.
            </p>
          </div>
        </div>

        {/* Timeline Bar */}
        <div className="timeline-interactive-bar">
          {place.timeline.map((ev, idx) => (
            <button
              key={`${ev.year}-${idx}`}
              type="button"
              className={`timeline-node-btn ${selectedEventIndex === idx ? 'active' : ''}`}
              onClick={() => setSelectedEventIndex(idx)}
            >
              <div className="node-marker" />
              <span className="node-year">{ev.year}</span>
              <span className="node-title">{ev.title.split(' ')[0]}</span>
            </button>
          ))}
        </div>

        {/* Selected Event Detail & Before/After Satellite Observation */}
        <div className="event-detail-container">
          <div className="event-header-row">
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <div className="event-date-chip">📅 {currentEvent.date}</div>
                {currentEvent.date === '2024-07-28' ? (
                  <span style={{ fontSize: '11px', background: 'rgba(56, 189, 248, 0.2)', color: '#38bdf8', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>
                    RECENT SATELLITE OBSERVATION
                  </span>
                ) : (
                  <span style={{ fontSize: '11px', background: 'rgba(148, 163, 184, 0.15)', color: '#94a3b8', padding: '2px 8px', borderRadius: 4 }}>
                    HISTORICAL ANALYSIS
                  </span>
                )}
              </div>
              <h3 className="event-headline">{currentEvent.title}</h3>
              <p className="event-summary">{currentEvent.impactSummary}</p>
            </div>
            <div className="event-sensor-pill">
              <span className="radar-beacon" />
              <span>{currentEvent.sensor}</span>
            </div>
          </div>

          {/* Before vs After Satellite Observation Card */}
          <div className="satellite-observation-grid">
            <div className="sat-panel before">
              <div className="panel-badge">1. PRE-FLOOD OBSERVATION</div>
              <div className="sat-image-placeholder">
                <div className="placeholder-water-overlay" />
                <div className="placeholder-content">
                  <span className="icon">🛰️</span>
                  <span>Sentinel-2 Optical (Pre-Monsoon)</span>
                  <small>Dry river channel &amp; healthy vegetative canopy</small>
                </div>
              </div>
              <div className="panel-caption">Pre-event baseline NDVI: 0.62</div>
            </div>

            <div className="sat-panel after">
              <div className="panel-badge highlight-cyan">2. POST-FLOOD SATELLITE EXTENT</div>
              <div className="sat-image-placeholder active-flood">
                <div className="placeholder-flood-water" />
                <div className="placeholder-content">
                  <span className="icon">🌊</span>
                  <span>Sentinel-1 Radar + Sentinel-2</span>
                  <small>Specular microwave reflection detecting open flood water</small>
                </div>
              </div>
              <div className="panel-caption">
                Observed Inundation Spread: {currentEvent.inundatedAreaKm2?.toLocaleString()} km²
              </div>
            </div>
          </div>

          {/* Change & Impact Metrics Bar */}
          <div className="change-analysis-metrics-bar">
            <div className="change-metric-box">
              <span className="box-label">Inundated Water Spread</span>
              <span className="box-val cyan">
                {currentEvent.inundatedAreaKm2 != null ? (
                  <>+{currentEvent.inundatedAreaKm2.toLocaleString()} <small>km²</small></>
                ) : (
                  <small style={{ fontSize: '13px', color: 'var(--fg-dim)' }}>Data not currently available</small>
                )}
              </span>
              <span className="box-sub">
                {currentEvent.inundatedAreaKm2 != null ? 'Projected metric UTM-45N' : 'Historical sensor archive'}
              </span>
            </div>

            <div className="change-metric-box">
              <span className="box-label">Vegetation Index Change (ΔNDVI)</span>
              <span className="box-val amber">
                {currentEvent.vegetationChangePct != null ? (
                  `${currentEvent.vegetationChangePct}%`
                ) : (
                  <small style={{ fontSize: '13px', color: 'var(--fg-dim)' }}>Data not currently available</small>
                )}
              </span>
              <span className="box-sub">
                {currentEvent.vegetationChangePct != null
                  ? 'Spectral vegetation loss in river corridor'
                  : 'Requires dual-temporal cloud-free optical pair'}
              </span>
            </div>

            <div className="change-metric-box">
              <span className="box-label">Affected Cropland</span>
              <span className="box-val emerald">
                {currentEvent.croplandAffectedKm2 != null ? (
                  <>{currentEvent.croplandAffectedKm2.toLocaleString()} <small>km²</small></>
                ) : (
                  <small style={{ fontSize: '13px', color: 'var(--fg-dim)' }}>Data not currently available</small>
                )}
              </span>
              <span className="box-sub">
                {currentEvent.croplandAffectedKm2 != null
                  ? 'Submerged agricultural land'
                  : 'Cropland mask intersection unavailable'}
              </span>
            </div>

            <div className="change-metric-box">
              <span className="box-label">Building Footprint Exposure</span>
              <span className="box-val purple">
                {currentEvent.buildingsExposed != null ? (
                  currentEvent.buildingsExposed.toLocaleString()
                ) : (
                  <small style={{ fontSize: '13px', color: 'var(--fg-dim)' }}>Data not currently available</small>
                )}
              </span>
              <span className="box-sub">
                {currentEvent.buildingsExposed != null
                  ? 'Structures in flood perimeter'
                  : 'Footprint registry unavailable for date'}
              </span>
            </div>

            <div className="change-metric-box">
              <span className="box-label">Road Network Exposure</span>
              <span className="box-val orange">
                {currentEvent.roadsExposedKm != null ? (
                  <>{currentEvent.roadsExposedKm.toLocaleString()} <small>km</small></>
                ) : (
                  <small style={{ fontSize: '13px', color: 'var(--fg-dim)' }}>Data not currently available</small>
                )}
              </span>
              <span className="box-sub">
                {currentEvent.roadsExposedKm != null
                  ? 'Highways & arterial roads'
                  : 'Road vector intersection unavailable'}
              </span>
            </div>
          </div>

          {/* Scientific Disclaimer Note */}
          <div className="scientific-disclaimer-box">
            <strong>Scientific Notice on Change Detection: </strong>
            A reduction in vegetation index (ΔNDVI) indicates vegetation stress, submergence, seasonal harvesting, or cloud attenuation; it does not by itself prove permanent crop destruction. Similarly, building exposure represents structural footprint counts intersecting satellite-detected water, not structural collapse or monetary loss.
          </div>

          {/* Collapsible Evidence Panel */}
          <div className="evidence-panel-wrapper">
            <button
              type="button"
              className="evidence-toggle-btn"
              onClick={() => setEvidenceOpen(!evidenceOpen)}
            >
              <span>{evidenceOpen ? '▼ Hide Satellite Evidence & Provenance' : '▶ View Satellite Evidence & Provenance'}</span>
            </button>

            {evidenceOpen && (
              <div className="evidence-details-drawer">
                <div className="evidence-grid-2">
                  <div>
                    <strong>Data Source: </strong>
                    <span>{currentEvent.evidence.source}</span>
                  </div>
                  <div>
                    <strong>Spatial Resolution: </strong>
                    <span>{currentEvent.evidence.resolution}</span>
                  </div>
                  <div>
                    <strong>Acquisition Date: </strong>
                    <span>{currentEvent.evidence.acquisitionDate}</span>
                  </div>
                  <div>
                    <strong>Processing Method: </strong>
                    <span>{currentEvent.evidence.method}</span>
                  </div>
                </div>
                <div className="evidence-limits">
                  <strong>Limitations: </strong>
                  <ul>
                    {currentEvent.evidence.limitations.map((l, idx) => (
                      <li key={idx}>• {l}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
