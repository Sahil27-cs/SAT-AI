'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { askAgent, type ChatReply, answeredBy } from '@/lib/api';
import { FormattedAnswer } from '@/components/FormattedAnswer';
import {
  PLACES_DATABASE,
  SEARCH_INDEX,
  getPlaceProfile,
  type PlaceProfile,
  type HistoricalEvent,
} from '@/lib/places-data';
import BIHAR_DATA from '@/lib/bihar-impact-data.json';
import ALL_38_DISTRICTS_2022 from '@/lib/bihar-all-38-districts.json';
import ALL_38_DISTRICTS_2024 from '@/lib/bihar-2024-districts.json';

interface ChatMessage {
  id: string;
  role: 'user' | 'agent' | 'error';
  text: string;
  meta?: ChatReply;
}

type SortKey = 'flooded_km2' | 'flood_pct' | 'cropland_km2' | 'impact_index';

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [selectedPlaceId, setSelectedPlaceId] = useState('bihar');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<typeof SEARCH_INDEX>([]);
  const [selectedEventIndex, setSelectedEventIndex] = useState<number>(4); // Default to 2022 validated event
  const [selectedDistrict, setSelectedDistrict] = useState<string>('Muzaffarpur');
  const [sortKey, setSortKey] = useState<SortKey>('flooded_km2');
  const [districtFilter, setDistrictFilter] = useState('');
  const [showAll38, setShowAll38] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);

  // Map Layer Toggles
  const [activeLayers, setActiveLayers] = useState({
    satelliteObs: true,
    vegetation: true,
    cropland: true,
    waterBodies: true,
    terrain: false,
    settlements: false,
    roads: false,
    floodExtent: true,
    historicalHazard: true,
  });

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const place: PlaceProfile = getPlaceProfile(selectedPlaceId);
  const currentEvent: HistoricalEvent =
    place.timeline[selectedEventIndex] || place.timeline[place.timeline.length - 1] || place.timeline[0];

  const isRecentEvent = currentEvent.date.startsWith('2024');

  // Toggle map layers
  const toggleLayer = (layer: keyof typeof activeLayers) => {
    setActiveLayers((prev) => ({ ...prev, [layer]: !prev[layer] }));
  };

  // Scroll to anchor
  const scrollToSection = (id: string) => {
    setMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Place search handling
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
    setSelectedEventIndex(id === 'bihar' ? 4 : 0);
  };

  // Active district data
  const districtData =
    ALL_38_DISTRICTS_2022.find((d) => d.district_name.toLowerCase() === selectedDistrict.toLowerCase()) ||
    ALL_38_DISTRICTS_2022[0];

  // 38 Districts list for active event
  const currentDistrictsRaw = isRecentEvent ? ALL_38_DISTRICTS_2024 : ALL_38_DISTRICTS_2022;
  const normalizedDistricts = currentDistrictsRaw.map((d: any) => ({
    name: d.district_name,
    flooded_km2: d.flooded_area_km2 || 0.0,
    district_area_km2: d.district_area_km2,
    flood_pct: d.flood_percentage || (d.flood_fraction ? d.flood_fraction * 100 : 0.0),
    cropland_km2: d.cropland_affected_km2 || d.cropland_exposed_km2 || 0.0,
    buildings: d.building_exposure ?? d.buildings_exposed ?? null,
    roads_km: d.road_exposure_km ?? d.roads_exposed_km ?? null,
    hazard_class: d.historical_hazard_class || d.historical_hazard || 'Moderate',
    basin: d.primary_river_basin || 'Ganga Basin',
    impact_index: d.sat_ai_impact_index ?? null,
  }));

  const filteredDistricts = normalizedDistricts
    .filter((d) => {
      if (districtFilter.trim()) {
        return d.name.toLowerCase().includes(districtFilter.toLowerCase().trim());
      }
      if (!showAll38) {
        return d.flooded_km2 > 0;
      }
      return true;
    })
    .sort((a, b) => {
      if (sortKey === 'flooded_km2') return b.flooded_km2 - a.flooded_km2;
      if (sortKey === 'flood_pct') return b.flood_pct - a.flood_pct;
      if (sortKey === 'cropland_km2') return b.cropland_km2 - a.cropland_km2;
      if (sortKey === 'impact_index') {
        const valA = a.impact_index ?? -1;
        const valB = b.impact_index ?? -1;
        return valB - valA;
      }
      return 0;
    });

  // Chat message sending
  const handleSendQuestion = useCallback(
    async (textToSend: string) => {
      const q = textToSend.trim();
      if (!q || isAsking) return;

      const userMsgId = `user-${Date.now()}`;
      setMessages((prev) => [...prev, { id: userMsgId, role: 'user', text: q }]);
      setInputQuestion('');
      setIsAsking(true);

      try {
        const reply = await askAgent(q, null);
        const agentMsgId = `agent-${Date.now()}`;

        // Ensure user-facing response is natural language and never raw telemetry unless requested
        const isAskingRaw = ['raw', 'telemetry', 'tool output', 'show evidence'].some((k) =>
          q.toLowerCase().includes(k)
        );
        let cleanText = reply.answer;
        if (!isAskingRaw && reply.answer.includes('### Verified Tool Telemetry')) {
          cleanText = reply.answer.split(/### Verified Tool Telemetry|Verified Tool Telemetry/i)[0].trim();
        }
        cleanText = cleanText
          .replace(/^Language layer unavailable — returning verified tool output directly\.\s*/i, '')
          .replace(/^### Scientific Assessment & Key Findings\s*/i, '')
          .trim();

        setMessages((prev) => [
          ...prev,
          {
            id: agentMsgId,
            role: 'agent',
            text: cleanText || reply.answer,
            meta: reply,
          },
        ]);
      } catch (e) {
        const errorMsgId = `err-${Date.now()}`;
        const detail = e instanceof Error && e.message ? e.message : 'Please try again.';
        setMessages((prev) => [
          ...prev,
          {
            id: errorMsgId,
            role: 'error',
            text: `The assistant could not answer. ${detail}`,
          },
        ]);
      } finally {
        setIsAsking(false);
      }
    },
    [isAsking]
  );

  useEffect(() => {
    if (messages.length > 0) {
      chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isAsking]);

  const copyToClipboard = (id: string, text: string) => {
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 2000);
      })
      .catch(() => undefined);
  };

  return (
    <>
      {/* ====================================================================
          1. NAVBAR
          ==================================================================== */}
      <header className="site-header">
        <div className="site-container nav-container">
          <a
            href="#hero"
            className="brand-link"
            onClick={(e) => {
              e.preventDefault();
              scrollToSection('hero');
            }}
          >
            <div className="brand-icon">
              <span className="brand-icon-pulse" />
            </div>
            <div className="brand-text">
              <span className="brand-title">SAT-AI</span>
              <span className="brand-sub">Satellite Intelligence Platform</span>
            </div>
          </a>

          <button
            type="button"
            className="mobile-nav-toggle"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? '✕' : '☰'}
          </button>

          <nav>
            <ul className={`nav-links ${mobileMenuOpen ? 'mobile-open' : ''}`}>
              <li>
                <a
                  href="#explore"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('explore');
                  }}
                >
                  Explore
                </a>
              </li>
              <li>
                <a
                  href="#disasters"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('disasters');
                  }}
                >
                  Disasters
                </a>
              </li>
              <li>
                <a
                  href="#satellite-insights"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('satellite-insights');
                  }}
                >
                  Satellite Insights
                </a>
              </li>
              <li>
                <a
                  href="#assistant"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('assistant');
                  }}
                >
                  Ask SAT-AI
                </a>
              </li>
              <li>
                <Link href="/research" className="nav-link">
                  Research
                </Link>
              </li>
              <li>
                <button
                  type="button"
                  className="nav-cta"
                  onClick={() => scrollToSection('explore')}
                >
                  <span>Explore a Place</span>
                  <span>→</span>
                </button>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="site-container">
        {/* ====================================================================
            2. HERO SECTION
            ==================================================================== */}
        <section id="hero" className="hero-section">
          <div className="hero-grid">
            <div className="hero-content">
              <div className="hero-eyebrow">
                <span>✦</span>
                <span>SAT-AI // SATELLITE INTELLIGENCE PLATFORM</span>
              </div>
              <h1 className="hero-title">
                Understand Any Place<br />From Space.
              </h1>
              <p className="hero-desc">
                Explore the environment, vegetation, water, terrain, disaster history, satellite observations, and spatial changes of any place.
              </p>
              <div className="hero-secondary-line">
                Search a place. Explore its story. Ask SAT-AI.
              </div>

              {/* Main Search Input */}
              <div className="hero-search-wrapper">
                <form
                  className="hero-search-bar"
                  onSubmit={(e) => {
                    e.preventDefault();
                    scrollToSection('explore');
                  }}
                >
                  <span className="hero-search-icon">🔍</span>
                  <input
                    type="text"
                    className="hero-search-input"
                    placeholder="Search a place, district or region (e.g. Bihar, Muzaffarpur, Mumbai, Assam)..."
                    value={searchQuery}
                    onChange={(e) => handleSearchChange(e.target.value)}
                  />
                  <button type="submit" className="quick-chip active" style={{ padding: '6px 16px' }}>
                    Look Up
                  </button>
                </form>

                {/* Autocomplete Dropdown in Hero if Typing */}
                {searchResults.length > 0 && (
                  <div className="place-search-dropdown" style={{ marginTop: '8px' }}>
                    {searchResults.map((item) => (
                      <button
                        key={`${item.id}-${item.name}`}
                        type="button"
                        className="dropdown-item"
                        onClick={() => {
                          handleSelectPlace(item.id, item.name);
                          scrollToSection('explore');
                        }}
                      >
                        <span className="item-name">{item.name}</span>
                        <span className="item-meta">
                          {item.type} • {item.parent}
                        </span>
                        {item.isAvailable && <span className="item-badge">Active Layers</span>}
                      </button>
                    ))}
                  </div>
                )}

                <div className="hero-quick-tags">
                  <span>Quick Explore:</span>
                  {[
                    { id: 'bihar', name: 'Bihar' },
                    { id: 'muzaffarpur', name: 'Muzaffarpur' },
                    { id: 'darbhanga', name: 'Darbhanga' },
                    { id: 'mumbai', name: 'Mumbai' },
                    { id: 'assam', name: 'Assam' },
                  ].map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      className={`quick-tag-chip ${selectedPlaceId === p.id ? 'active' : ''}`}
                      onClick={() => {
                        handleSelectPlace(p.id, p.name);
                        scrollToSection('explore');
                      }}
                    >
                      {p.name}
                    </button>
                  ))}
                </div>
              </div>

              <div className="hero-actions">
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => scrollToSection('explore')}
                >
                  <span>Explore a Place</span>
                  <span>→</span>
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    scrollToSection('assistant');
                    inputRef.current?.focus();
                  }}
                >
                  <span>Ask SAT-AI</span>
                </button>
              </div>
            </div>

            {/* Right Visual: Satellite Intelligence Platform HUD */}
            <div className="hero-visual">
              <div className="satellite-hud-card">
                <div className="hud-header-bar">
                  <span className="hud-tag">
                    <span>🛰️</span> SATELLITE INTELLIGENCE // EARTH OBSERVATION
                  </span>
                  <span className="hud-live-beacon">ACTIVE SATELLITE LAYERS</span>
                </div>

                <div className="hud-image-viewport">
                  <Image
                    src="/images/satellite-hero.jpg"
                    alt="Satellite view of river basin and floodplains from space"
                    width={1920}
                    height={1080}
                    priority
                  />
                  <div className="hud-overlay-grid" />
                </div>

                <div className="hud-telemetry-banner">
                  <div className="hud-telemetry-item">
                    <span className="hud-telemetry-label">Active Focus</span>
                    <span className="hud-telemetry-val">Ganga Floodplain (Bihar, India)</span>
                  </div>
                  <div className="hud-telemetry-item">
                    <span className="hud-telemetry-label">Constellation</span>
                    <span className="hud-telemetry-val">Sentinel-1 C-SAR &amp; Sentinel-2 MSI</span>
                  </div>
                  <div className="hud-telemetry-item">
                    <span className="hud-telemetry-label">Vegetation Dynamic</span>
                    <span className="hud-telemetry-val">NDVI 0.28 – 0.78 Monsoonal Cycle</span>
                  </div>
                  <div className="hud-telemetry-item">
                    <span className="hud-telemetry-label">Spatial Resolution</span>
                    <span className="hud-telemetry-val">10 m Metric Projected (UTM-45N)</span>
                  </div>
                </div>

                <div className="hud-footer-action">
                  <span>Explore Bihar through environmental and disaster layers</span>
                  <a
                    href="#explore"
                    onClick={(e) => {
                      e.preventDefault();
                      scrollToSection('explore');
                    }}
                  >
                    Inspect Place Profile ↓
                  </a>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ====================================================================
            3. ONE PLACE. MANY LAYERS OF INTELLIGENCE.
            ==================================================================== */}
        <section id="layers-of-intelligence" className="section-padding">
          <div className="section-header centered">
            <div className="section-tag">Platform Architecture</div>
            <h2 className="section-title">
              One Place. <span className="gradient-text">Many Layers of Intelligence.</span>
            </h2>
            <p className="section-lead">
              SAT-AI combines geographic, environmental and satellite information to build a richer picture of a place.
            </p>
          </div>

          <div className="flow-track-container">
            <div className="flow-step-node">
              <span className="icon">📍</span>
              <span className="name">Location</span>
              <span className="desc">Admin &amp; Coordinates</span>
            </div>
            <span className="flow-arrow">→</span>

            <div className="flow-step-node">
              <span className="icon">🌿</span>
              <span className="name">Environment</span>
              <span className="desc">Terrain, Crops &amp; Water</span>
            </div>
            <span className="flow-arrow">→</span>

            <div className="flow-step-node">
              <span className="icon">🛰️</span>
              <span className="name">Satellite</span>
              <span className="desc">Sentinel-1 &amp; Sentinel-2</span>
            </div>
            <span className="flow-arrow">→</span>

            <div className="flow-step-node">
              <span className="icon">🌊</span>
              <span className="name">Disasters</span>
              <span className="desc">Decadal Hazard History</span>
            </div>
            <span className="flow-arrow">→</span>

            <div className="flow-step-node">
              <span className="icon">🔄</span>
              <span className="name">Change</span>
              <span className="desc">Inundation &amp; ΔNDVI</span>
            </div>
            <span className="flow-arrow">→</span>

            <div className="flow-step-node">
              <span className="icon">🗺️</span>
              <span className="name">Impact</span>
              <span className="desc">38-District Exposure</span>
            </div>
            <span className="flow-arrow">→</span>

            <div className="flow-step-node">
              <span className="icon">🤖</span>
              <span className="name">AI Explanation</span>
              <span className="desc">Grounded Geospatial Copilot</span>
            </div>
          </div>

          <div className="intelligence-features-strip">
            <div className="feature-pill-node">🌿 Vegetation</div>
            <div className="feature-pill-node">🌾 Agriculture</div>
            <div className="feature-pill-node">💧 Water</div>
            <div className="feature-pill-node">🏔 Terrain</div>
            <div className="feature-pill-node">🏙 Built Environment</div>
            <div className="feature-pill-node">🌊 Flood</div>
            <div className="feature-pill-node">🔥 Fire</div>
            <div className="feature-pill-node">🌪 Cyclone</div>
            <div className="feature-pill-node">🌍 Earthquake</div>
            <div className="feature-pill-node">🛰 Satellite Change</div>
            <div className="feature-pill-node">🤖 AI Assistant</div>
          </div>
        </section>

        {/* ====================================================================
            4. EXPLORE A PLACE
            ==================================================================== */}
        <section id="explore" className="section-padding">
          <div className="section-header centered">
            <div className="section-tag">Geospatial Place Explorer</div>
            <h2 className="section-title">
              Explore a <span className="gradient-text">Place</span>
            </h2>
            <p className="section-lead">
              Search a place, district or region to inspect its environmental profile, historical hazards, and satellite observations.
            </p>

            <div className="place-search-container" style={{ margin: '24px auto 0', maxWidth: '640px' }}>
              <div className="place-search-bar">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  className="place-search-input"
                  placeholder="Search Bihar, Mumbai, Muzaffarpur, Darbhanga, Assam..."
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
                      {item.isAvailable && <span className="item-badge">Active Layers</span>}
                    </button>
                  ))}
                </div>
              )}

              <div className="quick-place-chips" style={{ justifyContent: 'center', marginTop: '14px' }}>
                <span className="chips-label">Quick Places:</span>
                {[
                  { id: 'bihar', label: 'Bihar' },
                  { id: 'muzaffarpur', label: 'Muzaffarpur' },
                  { id: 'darbhanga', label: 'Darbhanga' },
                  { id: 'mumbai', label: 'Mumbai' },
                  { id: 'assam', label: 'Assam' },
                ].map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    className={`quick-chip ${selectedPlaceId === p.id ? 'active' : ''}`}
                    onClick={() => handleSelectPlace(p.id, p.label)}
                  >
                    📍 {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Place Profile Banner */}
          <div className="place-header-banner" style={{ marginTop: '28px' }}>
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
              5. ENVIRONMENTAL PROFILE
              ==================================================================== */}
          <div id="environmental-profile" style={{ marginTop: '48px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">🌿</span>
              <div>
                <h3 className="sub-title">Environmental Profile</h3>
                <p className="sub-desc">
                  Core physical and land-cover baseline characteristics for {place.name}.
                </p>
              </div>
            </div>

            <div className="environment-cards-grid">
              {/* 1. VEGETATION */}
              <div className="env-card">
                <div className="env-card-header">
                  <span className="env-icon">🌿</span>
                  <h4>{place.environment.vegetation.title}</h4>
                </div>
                <div className="env-metrics">
                  <div className="env-metric-item">
                    <span className="label">Forest Cover</span>
                    <span className="val cyan">
                      {place.environment.vegetation.forestCoverKm2 != null ? (
                        <>{place.environment.vegetation.forestCoverKm2.toLocaleString()} km²</>
                      ) : (
                        'Data currently unavailable'
                      )}
                    </span>
                  </div>
                  <div className="env-metric-item">
                    <span className="label">Canopy Cover</span>
                    <span className="val">
                      {place.environment.vegetation.forestCoverPct != null ? (
                        <>{place.environment.vegetation.forestCoverPct}%</>
                      ) : (
                        'Data currently unavailable'
                      )}
                    </span>
                  </div>
                </div>
                <div className="env-detail-block">
                  <small>Dominant Vegetation Types:</small>
                  <ul>
                    {place.environment.vegetation.dominantTypes.map((t, idx) => (
                      <li key={idx}>{t}</li>
                    ))}
                  </ul>
                </div>
                <div className="env-detail-block">
                  <small>Seasonal NDVI Dynamics:</small>
                  <p>{place.environment.vegetation.seasonalDynamics}</p>
                </div>
                <div className="env-source">Source: {place.environment.vegetation.source}</div>
              </div>

              {/* 2. AGRICULTURE */}
              <div className="env-card">
                <div className="env-card-header">
                  <span className="env-icon">🌾</span>
                  <h4>{place.environment.agriculture.title}</h4>
                </div>
                <div className="env-metrics">
                  <div className="env-metric-item">
                    <span className="label">Net Cropped Area</span>
                    <span className="val emerald">
                      {place.environment.agriculture.netCroppedAreaKm2 != null ? (
                        <>{place.environment.agriculture.netCroppedAreaKm2.toLocaleString()} km²</>
                      ) : (
                        'Data currently unavailable'
                      )}
                    </span>
                  </div>
                  <div className="env-metric-item">
                    <span className="label">Cropland Fraction</span>
                    <span className="val">
                      {place.environment.agriculture.croppedAreaPct != null ? (
                        <>{place.environment.agriculture.croppedAreaPct}%</>
                      ) : (
                        'Data currently unavailable'
                      )}
                    </span>
                  </div>
                </div>
                <div className="env-detail-block">
                  <small>Major Agricultural Crops:</small>
                  <div className="crops-tags">
                    {place.environment.agriculture.majorCrops.map((c, idx) => (
                      <span key={idx} className="crop-tag">
                        {c}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="env-detail-block">
                  <small>Floodplain Farming &amp; Cropping:</small>
                  <p>{place.environment.agriculture.floodplainFarming}</p>
                </div>
                <div className="env-source">Source: {place.environment.agriculture.source}</div>
              </div>

              {/* 3. WATER */}
              <div className="env-card">
                <div className="env-card-header">
                  <span className="env-icon">💧</span>
                  <h4>{place.environment.water.title}</h4>
                </div>
                <div className="env-detail-block">
                  <small>Major River Systems:</small>
                  <div className="rivers-tags">
                    {place.environment.water.majorRivers.map((r, idx) => (
                      <span key={idx} className="river-tag">
                        {r}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="env-detail-block">
                  <small>Basin Characteristics:</small>
                  <p>{place.environment.water.basinDescription}</p>
                </div>
                <div className="env-detail-block">
                  <small>Seasonal Wetlands:</small>
                  <p>{place.environment.water.seasonalWaterBodies}</p>
                </div>
                <div className="env-source">Source: {place.environment.water.source}</div>
              </div>

              {/* 4. TERRAIN */}
              <div className="env-card">
                <div className="env-card-header">
                  <span className="env-icon">🏔️</span>
                  <h4>{place.environment.terrain.title}</h4>
                </div>
                <div className="env-metrics">
                  <div className="env-metric-item">
                    <span className="label">Elevation Range</span>
                    <span className="val amber">{place.environment.terrain.elevationRangeM}</span>
                  </div>
                  <div className="env-metric-item">
                    <span className="label">Average Slope</span>
                    <span className="val">{place.environment.terrain.averageSlope}</span>
                  </div>
                </div>
                <div className="env-detail-block">
                  <small>Geomorphology:</small>
                  <p>{place.environment.terrain.geomorphology}</p>
                </div>
                <div className="env-detail-block">
                  <small>Drainage Pattern:</small>
                  <p>{place.environment.terrain.drainagePattern}</p>
                </div>
                <div className="env-source">Source: {place.environment.terrain.source}</div>
              </div>

              {/* 5. BUILT ENVIRONMENT */}
              <div className="env-card">
                <div className="env-card-header">
                  <span className="env-icon">🏙️</span>
                  <h4>{place.environment.builtEnvironment.title}</h4>
                </div>
                <div className="env-metrics">
                  <div className="env-metric-item">
                    <span className="label">Settlements</span>
                    <span className="val purple">{place.environment.builtEnvironment.settlementCount}</span>
                  </div>
                  <div className="env-metric-item">
                    <span className="label">Density</span>
                    <span className="val">{place.environment.builtEnvironment.settlementDensity}</span>
                  </div>
                </div>
                <div className="env-detail-block">
                  <small>Major Arterial Roadways:</small>
                  <p>
                    {place.environment.builtEnvironment.arterialRoadsKm != null
                      ? `${place.environment.builtEnvironment.arterialRoadsKm.toLocaleString()} km of arterial road alignments.`
                      : 'Data currently unavailable.'}
                  </p>
                </div>
                <div className="env-source">Source: {place.environment.builtEnvironment.source}</div>
              </div>
            </div>
          </div>

          {/* ====================================================================
              6. LANDSCAPE MAP
              ==================================================================== */}
          <div id="landscape-map" style={{ marginTop: '56px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">🗺️</span>
              <div>
                <h3 className="sub-title">Landscape &amp; Multi-Layer Map</h3>
                <p className="sub-desc">
                  Interactive multi-layer geospatial inspection of river basins, croplands, and observed flood inundation across Bihar.
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

              {/* Interactive SVG Geospatial Canvas */}
              <div className="map-canvas-container">
                <svg
                  className="interactive-svg-canvas"
                  viewBox="0 0 900 520"
                  preserveAspectRatio="xMidYMid meet"
                >
                  <rect x="0" y="0" width="900" height="520" fill="#030712" />

                  {/* Rivers */}
                  {activeLayers.waterBodies && (
                    <g className="rivers-layer">
                      <path
                        d="M 50 310 Q 250 320, 420 300 T 650 310 T 850 330"
                        fill="none"
                        stroke="#0284c7"
                        strokeWidth="7"
                        opacity="0.6"
                      />
                      <path
                        d="M 160 50 Q 240 180, 410 300"
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="3.5"
                        opacity="0.5"
                      />
                      <path
                        d="M 270 50 Q 340 160, 520 305"
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="3.5"
                        opacity="0.5"
                      />
                      <path
                        d="M 370 40 Q 420 140, 560 305"
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="3"
                        opacity="0.5"
                      />
                      <path
                        d="M 580 40 Q 610 160, 680 315"
                        fill="none"
                        stroke="#38bdf8"
                        strokeWidth="5"
                        opacity="0.6"
                      />
                    </g>
                  )}

                  {/* District Polygons */}
                  <g className="districts-polygons">
                    {ALL_38_DISTRICTS_2022.map((d, idx) => {
                      const isSelected = selectedDistrict.toLowerCase() === d.district_name.toLowerCase();
                      const col = idx % 7;
                      const row = Math.floor(idx / 7);
                      const x = 75 + col * 110 + (row % 2 === 1 ? 25 : 0);
                      const y = 80 + row * 90;

                      let fillColor = '#0f172a';
                      if (activeLayers.floodExtent && d.flood_percentage > 10) {
                        fillColor = 'rgba(56, 189, 248, 0.32)';
                      } else if (activeLayers.cropland) {
                        fillColor = 'rgba(16, 185, 129, 0.12)';
                      } else if (activeLayers.historicalHazard && d.historical_hazard_class === 'Very High') {
                        fillColor = 'rgba(244, 63, 94, 0.2)';
                      }

                      return (
                        <g
                          key={d.district_name}
                          className="district-map-node"
                          onClick={() => setSelectedDistrict(d.district_name)}
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
                            {d.district_name}
                          </text>
                          {activeLayers.floodExtent && d.flood_percentage > 5 && (
                            <text x={x} y={y - 8} className="district-svg-pct">
                              {d.flood_percentage}%
                            </text>
                          )}
                        </g>
                      );
                    })}
                  </g>
                </svg>

                {/* District Inspector Card Overlay */}
                <div className="map-district-inspector">
                  <div className="inspector-header">
                    <span className="inspector-badge">Selected District</span>
                    <h4 className="inspector-name">{districtData.district_name}</h4>
                    <span className="inspector-basin">Basin: {districtData.primary_river_basin}</span>
                  </div>

                  <div className="inspector-metrics">
                    <div className="insp-row">
                      <span>Flooded Extent:</span>
                      <strong className="cyan">
                        {districtData.flooded_area_km2} km² ({districtData.flood_percentage}%)
                      </strong>
                    </div>
                    <div className="insp-row">
                      <span>Cropland Affected:</span>
                      <strong className="emerald">{districtData.cropland_affected_km2} km²</strong>
                    </div>
                    <div className="insp-row">
                      <span>Building Structures:</span>
                      <strong>
                        {districtData.building_exposure != null
                          ? districtData.building_exposure.toLocaleString()
                          : 'Data unavailable'}
                      </strong>
                    </div>
                    <div className="insp-row">
                      <span>Arterial Roads:</span>
                      <strong>
                        {districtData.road_exposure_km != null
                          ? `${districtData.road_exposure_km} km`
                          : 'Data unavailable'}
                      </strong>
                    </div>
                    <div className="insp-row">
                      <span>Historical Hazard:</span>
                      <span
                        className={`hazard-chip ${districtData.historical_hazard_class
                          .toLowerCase()
                          .replace(' ', '-')}`}
                      >
                        {districtData.historical_hazard_class}
                      </span>
                    </div>
                    <div className="insp-row highlight-box">
                      <span>SAT-AI Impact Index:</span>
                      <strong className="cyan">
                        {districtData.sat_ai_impact_index != null
                          ? districtData.sat_ai_impact_index
                          : 'Uncalculated'}
                      </strong>
                    </div>
                  </div>

                  <div className="inspector-blocks">
                    <small>Inundated Blocks Inside Footprint:</small>
                    <div className="blocks-list">
                      {districtData.blocks_affected.map((b, idx) => (
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
              7. DISASTER PROFILE
              ==================================================================== */}
          <div id="disasters" style={{ marginTop: '56px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">⚡</span>
              <div>
                <h3 className="sub-title">Disaster Profile</h3>
                <p className="sub-desc">
                  Decadal hazard presence and historical frequencies backed by authoritative disaster registries.
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
              8. DISASTER HISTORY TIMELINE
              ==================================================================== */}
          <div id="timeline" style={{ marginTop: '56px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">📅</span>
              <div>
                <h3 className="sub-title">Disaster History</h3>
                <p className="sub-desc">
                  Explore verified historical flood events and recent satellite acquisitions. Select any event to inspect its satellite observations.
                </p>
              </div>
            </div>

            {/* Timeline Horizontal Nodes */}
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
          </div>

          {/* ====================================================================
              9. SATELLITE INSIGHTS
              ==================================================================== */}
          <div id="satellite-insights" style={{ marginTop: '48px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">🛰️</span>
              <div>
                <h3 className="sub-title">Satellite Insight</h3>
                <p className="sub-desc">
                  Dual-temporal satellite observation before and after the flood event over {place.name}.
                </p>
              </div>
            </div>

            <div className="event-detail-container">
              {/* Event Header with Status Badge */}
              <div className="event-header-row">
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <div className="event-date-chip">📅 {currentEvent.date}</div>
                    {isRecentEvent ? (
                      <span
                        style={{
                          fontSize: '11px',
                          background: 'rgba(56, 189, 248, 0.2)',
                          color: '#38bdf8',
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontWeight: 700,
                        }}
                      >
                        RECENT SATELLITE OBSERVATION
                      </span>
                    ) : (
                      <span
                        style={{
                          fontSize: '11px',
                          background: 'rgba(16, 185, 129, 0.15)',
                          color: '#10b981',
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontWeight: 700,
                        }}
                      >
                        HISTORICAL ANALYSIS (VALIDATED BENCHMARK)
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

              {/* Event Switcher Notice */}
              {isRecentEvent ? (
                <div
                  style={{
                    margin: '16px 0',
                    padding: '12px 16px',
                    background: 'rgba(56, 189, 248, 0.08)',
                    border: '1px solid rgba(56, 189, 248, 0.25)',
                    borderRadius: 8,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: 12,
                  }}
                >
                  <p style={{ margin: 0, fontSize: '12.5px', color: '#cbd5e1' }}>
                    <strong>Observation Date:</strong> 28 July 2024 • Status:{' '}
                    <strong style={{ color: '#38bdf8' }}>MODEL-INFERRED FLOOD EXTENT</strong> (Passed radiometric distribution gate; unground-truthed recent pass).
                  </p>
                  <button
                    type="button"
                    className="text-btn"
                    style={{
                      fontSize: '12px',
                      color: '#38bdf8',
                      background: 'transparent',
                      border: '1px solid #38bdf8',
                      borderRadius: 4,
                      padding: '4px 10px',
                      cursor: 'pointer',
                    }}
                    onClick={() => setSelectedEventIndex(4)}
                  >
                    ⇄ Compare with 2022 Historical Baseline
                  </button>
                </div>
              ) : (
                <div
                  style={{
                    margin: '16px 0',
                    padding: '12px 16px',
                    background: 'rgba(16, 185, 129, 0.08)',
                    border: '1px solid rgba(16, 185, 129, 0.25)',
                    borderRadius: 8,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    flexWrap: 'wrap',
                    gap: 12,
                  }}
                >
                  <p style={{ margin: 0, fontSize: '12.5px', color: '#cbd5e1' }}>
                    <strong>Observation Date:</strong> 15 October 2022 • Status:{' '}
                    <strong style={{ color: '#10b981' }}>VALIDATED BASELINE BENCHMARK</strong> (Includes BFCD-22 ground-truth crop damage survey in Muzaffarpur).
                  </p>
                  <button
                    type="button"
                    className="text-btn"
                    style={{
                      fontSize: '12px',
                      color: '#10b981',
                      background: 'transparent',
                      border: '1px solid #10b981',
                      borderRadius: 4,
                      padding: '4px 10px',
                      cursor: 'pointer',
                    }}
                    onClick={() => setSelectedEventIndex(5)}
                  >
                    ⇄ Inspect 2024 Recent Satellite Scene
                  </button>
                </div>
              )}

              {/* Dual-Temporal High-Res Satellite Observation Panels */}
              <div className="satellite-observation-grid">
                <div className="sat-panel before">
                  <div className="panel-badge">1. PRE-FLOOD OBSERVATION</div>
                  <div className="sat-image-placeholder">
                    <Image
                      src="/images/hero-flood-before.jpg"
                      alt="Pre-flood Sentinel-2 natural color optical capture showing normal river channel"
                      width={1280}
                      height={720}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  </div>
                  <div className="panel-caption">
                    Pre-event baseline NDVI: 0.64 • Normal low-flow river channel &amp; green crop parcels
                  </div>
                </div>

                <div className="sat-panel after">
                  <div className="panel-badge highlight-cyan">2. POST-FLOOD SATELLITE EXTENT</div>
                  <div className="sat-image-placeholder active-flood">
                    <Image
                      src="/images/hero-flood-after.jpg"
                      alt="Post-flood Sentinel-1 microwave radar and optical capture showing inundation"
                      width={1280}
                      height={720}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                  </div>
                  <div className="panel-caption">
                    Observed Inundation Spread: {currentEvent.inundatedAreaKm2?.toLocaleString()} km² • Specular microwave radar reflection
                  </div>
                </div>
              </div>

              {/* Satellite Metrics Bar */}
              <div className="change-analysis-metrics-bar">
                <div className="change-metric-box">
                  <span className="box-label">Observed Inundation</span>
                  <span className="box-val cyan">
                    {currentEvent.inundatedAreaKm2 != null ? (
                      <>+{currentEvent.inundatedAreaKm2.toLocaleString()} <small>km²</small></>
                    ) : (
                      'Data unavailable'
                    )}
                  </span>
                  <span className="box-sub">
                    {currentEvent.inundatedAreaKm2 != null
                      ? `${((currentEvent.inundatedAreaKm2 / place.totalAreaKm2) * 100).toFixed(2)}% of analyzed area`
                      : 'Observation unmeasured'}
                  </span>
                </div>

                <div className="change-metric-box">
                  <span className="box-label">Vegetation Change</span>
                  <span className="box-val amber">
                    {currentEvent.vegetationChangePct != null
                      ? `${currentEvent.vegetationChangePct}% ΔNDVI`
                      : 'Data unavailable'}
                  </span>
                  <span className="box-sub">Spectral attenuation in flood corridor</span>
                </div>

                <div className="change-metric-box">
                  <span className="box-label">Affected Cropland</span>
                  <span className="box-val emerald">
                    {currentEvent.croplandAffectedKm2 != null ? (
                      <>{currentEvent.croplandAffectedKm2.toLocaleString()} <small>km²</small></>
                    ) : (
                      'Data unavailable'
                    )}
                  </span>
                  <span className="box-sub">Cropland intersecting inundation footprint</span>
                </div>

                <div className="change-metric-box">
                  <span className="box-label">Observation Sensor</span>
                  <span className="box-val" style={{ fontSize: '15px' }}>
                    Sentinel-1 C-SAR
                  </span>
                  <span className="box-sub">10m Ground Range Detected (GRD)</span>
                </div>
              </div>
            </div>
          </div>

          {/* ====================================================================
              10. WHAT CHANGED? (CHANGE ANALYSIS)
              ==================================================================== */}
          <div id="what-changed" style={{ marginTop: '56px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">🔄</span>
              <div>
                <h3 className="sub-title">What Changed?</h3>
                <p className="sub-desc">
                  Comparative change detection between pre-flood baseline and the satellite-observed flood pulse.
                </p>
              </div>
            </div>

            <div className="change-analysis-grid">
              <div className="change-analysis-card">
                <div className="change-card-icon">💧</div>
                <div className="change-card-title">Water Surface Spread</div>
                <div className="change-card-metric cyan">
                  {currentEvent.inundatedAreaKm2 != null ? (
                    `+${currentEvent.inundatedAreaKm2.toLocaleString()} km²`
                  ) : (
                    'Unmeasured'
                  )}
                </div>
                <div className="change-card-comparison">
                  {currentEvent.inundatedAreaKm2 != null
                    ? `Expanded from normal dry-season river mainstem (~1,850 km²) to state-wide floodwaters.`
                    : 'Inundation footprint unmeasured for this historical overpass.'}
                </div>
              </div>

              <div className="change-analysis-card">
                <div className="change-card-icon">🌿</div>
                <div className="change-card-title">Vegetation Index Signal</div>
                <div className="change-card-metric amber">
                  {currentEvent.vegetationChangePct != null
                    ? `${currentEvent.vegetationChangePct}%`
                    : 'Unmeasured'}
                </div>
                <div className="change-card-comparison">
                  {currentEvent.vegetationChangePct != null
                    ? `Observed mean NDVI change across flooded corridor between pre- and post-event passes.`
                    : 'Dual-temporal multispectral surface reflectance not acquired for this event.'}
                </div>
              </div>

              <div className="change-analysis-card">
                <div className="change-card-icon">🌾</div>
                <div className="change-card-title">Cropland Exposure</div>
                <div className="change-card-metric emerald">
                  {currentEvent.croplandAffectedKm2 != null ? (
                    `${currentEvent.croplandAffectedKm2.toLocaleString()} km²`
                  ) : (
                    'Unmeasured'
                  )}
                </div>
                <div className="change-card-comparison">
                  {currentEvent.croplandAffectedKm2 != null
                    ? 'Agricultural fields intersecting detected standing water across river basins.'
                    : 'High-resolution agricultural crop masks were not recorded for this event.'}
                </div>
              </div>

              <div className="change-analysis-card">
                <div className="change-card-icon">🏙️</div>
                <div className="change-card-title">Settlement Exposure</div>
                <div className="change-card-metric purple">
                  {currentEvent.buildingsExposed != null ? (
                    `~${currentEvent.buildingsExposed.toLocaleString()}`
                  ) : (
                    'Unmeasured'
                  )}
                </div>
                <div className="change-card-comparison">
                  {currentEvent.buildingsExposed != null
                    ? 'Building footprint structures intersecting open floodwater boundary.'
                    : 'Cadastral settlement layer unmeasured for this observation.'}
                </div>
              </div>
            </div>
          </div>

          {/* ====================================================================
              11. VEGETATION & LAND CHANGE
              ==================================================================== */}
          <div id="vegetation-change" style={{ marginTop: '48px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">🌱</span>
              <div>
                <h3 className="sub-title">Vegetation &amp; Land Change</h3>
                <p className="sub-desc">
                  Dual-temporal Normalized Difference Vegetation Index (NDVI) measured via Sentinel-2 MSI multispectral surface reflectance.
                </p>
              </div>
            </div>

            {currentEvent.vegetationChangePct != null ? (
              <div className="ndvi-comparison-container">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
                  <span style={{ fontSize: '14.5px', fontWeight: 700, color: '#fff' }}>
                    Sentinel-2 Dual-Temporal NDVI Comparison — {currentEvent.title}
                  </span>
                  <span className="hazard-chip low" style={{ fontFamily: 'var(--mono)', fontSize: '11px' }}>
                    Bands: B4 (Red: 665nm) &amp; B8 (NIR: 842nm)
                  </span>
                </div>

                <div className="ndvi-chart-bars">
                  <div className="ndvi-bar-item">
                    <div className="ndvi-bar-meta">
                      <span>Pre-Flood Baseline NDVI</span>
                      <strong>{isRecentEvent ? '0.68 (July Baseline)' : '0.64 (Healthy Canopy)'}</strong>
                    </div>
                    <div className="ndvi-track">
                      <div className="ndvi-progress pre" style={{ width: isRecentEvent ? '68%' : '64%' }} />
                    </div>
                  </div>

                  <div className="ndvi-bar-item">
                    <div className="ndvi-bar-meta">
                      <span>Post-Flood Measured NDVI</span>
                      <strong>{isRecentEvent ? '0.55 (Attenuated)' : '0.46 (Attenuated Signal)'}</strong>
                    </div>
                    <div className="ndvi-track">
                      <div className="ndvi-progress post" style={{ width: isRecentEvent ? '55%' : '46%' }} />
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '16px', margin: '14px 0' }}>
                  <span style={{ fontSize: '13px', color: 'var(--fg-dim)' }}>Measured Relative Change:</span>
                  <strong style={{ fontSize: '15px', color: '#f59e0b', fontFamily: 'var(--mono)' }}>
                    {currentEvent.vegetationChangePct}% (ΔNDVI = {isRecentEvent ? '-0.13' : '-0.18'})
                  </strong>
                </div>

                <div className="scientific-disclaimer-box">
                  <strong>Important Scientific Context: </strong>
                  Vegetation-index change indicates a change in vegetation signal. It may be associated with inundation,
                  vegetation stress, seasonal change, harvesting or other land-cover changes.
                  <br />
                  <span style={{ color: '#f59e0b', fontWeight: 600 }}>
                    NDVI change does not by itself prove permanent crop destruction.
                  </span>
                </div>
              </div>
            ) : (
              <div className="ndvi-comparison-container" style={{ padding: '24px', textAlign: 'center' }}>
                <p style={{ color: 'var(--fg-dim)', margin: '0 0 8px 0', fontSize: '14px' }}>
                  📡 <strong>Optical NDVI comparison unavailable for {currentEvent.title} ({currentEvent.year})</strong>
                </p>
                <p style={{ color: 'var(--fg-faint)', margin: 0, fontSize: '12.5px' }}>
                  Dual-temporal optical imagery from Sentinel-2 MSI is available from 2015 onwards. Historical events (1998, 2004) rely on radar/optical archives without standardized surface reflectance pairs. In accordance with SAT-AI's Zero-Hallucination policy, synthetic placeholder numbers are strictly withheld.
                </p>
              </div>
            )}
          </div>

          {/* ====================================================================
              12. AGRICULTURE & EXPOSURE
              ==================================================================== */}
          <div id="agriculture-exposure" style={{ marginTop: '48px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">🌾</span>
              <div>
                <h3 className="sub-title">Agriculture &amp; Exposure</h3>
                <p className="sub-desc">
                  Evaluation of agricultural cropland intersecting observed inundation and ground-truth verified crop damage categories.
                </p>
              </div>
            </div>

            <div className="crop-damage-banner">
              <div className="crop-banner-header">
                <div className="crop-banner-title">
                  🌾 Cropland Damage Breakdown — {BIHAR_DATA.bfcd22_crop_damage.region} (BFCD-22 Ground-Truth Survey)
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
                BFCD-22 ground truth is validated specifically in Muzaffarpur. All other district figures strictly represent
                cropland intersecting observed inundation from ESA WorldCover, never unvalidated crop loss.
              </div>
            </div>
          </div>

          {/* ====================================================================
              13. WHERE WAS THE IMPACT? (DISTRICT IMPACT)
              ==================================================================== */}
          <div id="district-impact" style={{ marginTop: '56px' }}>
            <div className="sub-section-header">
              <span className="sub-icon">📊</span>
              <div>
                <h3 className="sub-title">Where Was the Impact?</h3>
                <p className="sub-desc">
                  Interactive geospatial aggregation across Bihar districts for the selected satellite observation. Click any row to inspect.
                </p>
              </div>
            </div>

            <div className="impact-table-card">
              <div className="table-header-row">
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                  <input
                    type="text"
                    placeholder="Filter district..."
                    value={districtFilter}
                    onChange={(e) => setDistrictFilter(e.target.value)}
                    style={{
                      background: 'rgba(255,255,255,0.05)',
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '6px 12px',
                      color: '#fff',
                      fontSize: '13px',
                      outline: 'none',
                    }}
                  />
                  <button
                    type="button"
                    className={`sort-btn ${!showAll38 ? 'active' : ''}`}
                    onClick={() => setShowAll38(false)}
                  >
                    Affected Districts Only ({normalizedDistricts.filter((d) => d.flooded_km2 > 0).length})
                  </button>
                  <button
                    type="button"
                    className={`sort-btn ${showAll38 ? 'active' : ''}`}
                    onClick={() => setShowAll38(true)}
                  >
                    All 38 Districts
                  </button>
                </div>

                <div className="sort-controls">
                  <span className="sort-label">Sort:</span>
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
                  {!isRecentEvent && (
                    <button
                      type="button"
                      className={`sort-btn ${sortKey === 'impact_index' ? 'active' : ''}`}
                      onClick={() => setSortKey('impact_index')}
                    >
                      Impact Index
                    </button>
                  )}
                </div>
              </div>

              <div className="table-responsive">
                <table className="bihar-impact-table">
                  <thead>
                    <tr>
                      <th>District</th>
                      <th>Observed Flood Area</th>
                      <th>Flooded %</th>
                      <th>Cropland Exposure</th>
                      <th>Buildings</th>
                      <th>Roads (km)</th>
                      <th>Historical Hazard</th>
                      {!isRecentEvent && <th>SAT-AI Impact Index</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDistricts.map((d) => (
                      <tr
                        key={d.name}
                        className={selectedDistrict.toLowerCase() === d.name.toLowerCase() ? 'highlight-row' : ''}
                        onClick={() => {
                          setSelectedDistrict(d.name);
                          scrollToSection('landscape-map');
                        }}
                        style={{ cursor: 'pointer' }}
                      >
                        <td className="district-name-cell">
                          <strong>{d.name}</strong>
                          <span style={{ display: 'block', fontSize: '11px', color: 'var(--fg-dim)' }}>
                            {d.basin}
                          </span>
                        </td>
                        <td>
                          <span className="mono-num cyan">
                            {d.flooded_km2 > 0 ? `${d.flooded_km2} km²` : '0.0 km²'}
                          </span>
                        </td>
                        <td>
                          <span className="mono-num">{d.flood_pct > 0 ? `${d.flood_pct}%` : '0.0%'}</span>
                        </td>
                        <td>
                          <span className="mono-num emerald">
                            {d.cropland_km2 > 0 ? `${d.cropland_km2} km²` : '0.0 km²'}
                          </span>
                        </td>
                        <td>
                          <span className="mono-num">
                            {d.buildings != null ? d.buildings.toLocaleString() : '—'}
                          </span>
                        </td>
                        <td>
                          <span className="mono-num">
                            {d.roads_km != null ? `${d.roads_km} km` : '—'}
                          </span>
                        </td>
                        <td>
                          <span className={`table-hazard-tag ${d.hazard_class.toLowerCase().replace(' ', '-')}`}>
                            {d.hazard_class}
                          </span>
                        </td>
                        {!isRecentEvent && (
                          <td>
                            <span className="impact-index-pill">{d.impact_index}</span>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Evidence & Scientific Provenance Drawer */}
              <div className="evidence-accordion-wrapper">
                <button
                  type="button"
                  className="toggle-evidence-btn"
                  onClick={() => setEvidenceOpen((prev) => !prev)}
                >
                  <span>
                    {evidenceOpen ? '▼ Hide Evidence & Scientific Provenance' : '▶ Show Evidence & Scientific Provenance'}
                  </span>
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
          </div>

          {/* ====================================================================
              14. ASK SAT-AI
              ==================================================================== */}
          <div id="assistant" style={{ marginTop: '56px' }}>
            <div className="section-header centered">
              <div className="section-tag">Interactive Geospatial Copilot</div>
              <h2 className="section-title">
                Ask <span className="gradient-text">SAT-AI</span>
              </h2>
              <p className="section-lead">
                Ask questions about {place.name}, its environment, disasters, and satellite observations.
              </p>
            </div>

            <div className="assistant-card">
              <div className="assistant-header">
                <div className="assistant-header-left">
                  <span className="assistant-status-beacon" />
                  <span className="assistant-header-title">SAT-AI Assistant — Context: {place.name}</span>
                  <span className="assistant-badge">GROUNDED COPILOT</span>
                </div>
                <div className="assistant-controls">
                  <span className="kbd-hint">Press Enter ↵</span>
                  {messages.length > 0 && (
                    <button
                      type="button"
                      className="clear-chat-btn"
                      onClick={() => setMessages([])}
                      title="Clear chat history"
                    >
                      <span>✕</span> Clear chat
                    </button>
                  )}
                </div>
              </div>

              {/* Place-specific starter questions */}
              <div className="starter-prompts">
                <div className="starter-prompts-label">Suggested Questions for {place.name}:</div>
                <div className="prompt-chips-grid">
                  {[
                    `What is the situation in ${place.name}?`,
                    'Which areas are affected by floods?',
                    'How much area was inundated?',
                    'How has vegetation changed?',
                    'How much cropland is exposed?',
                    `When does ${place.name} usually experience floods?`,
                    'What changed between pre-flood and post-flood imagery?',
                    'Which district was most affected?',
                    'Is there a recent satellite observation?',
                    'Is this an official government warning or SAT-AI analysis?',
                  ].map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="prompt-chip"
                      onClick={() => handleSendQuestion(q)}
                      disabled={isAsking}
                    >
                      <span>✦</span> {q}
                    </button>
                  ))}
                </div>
              </div>

              {/* Chat Messages Window */}
              <div className="chat-window">
                {messages.length === 0 && (
                  <div className="chat-empty-state">
                    <div className="chat-empty-icon">🛰️</div>
                    <div className="chat-empty-title">Satellite Intelligence Copilot Ready</div>
                    <div className="chat-empty-text">
                      Ask any question above about {place.name} flood inundation, affected districts,
                      cropland exposure, terrain, or satellite observation dates.
                    </div>
                  </div>
                )}

                {messages.map((m) => (
                  <div key={m.id} className={`chat-msg ${m.role}`}>
                    <div className="msg-sender">
                      <span className={`sender-badge ${m.role === 'user' ? 'user-badge' : ''}`}>
                        {m.role === 'user' ? '👤 You' : m.role === 'error' ? '⚠️ Notice' : '✦ SAT-AI Assistant'}
                      </span>
                      {m.role === 'agent' && (
                        <button
                          type="button"
                          className="copy-msg-btn"
                          onClick={() => copyToClipboard(m.id, m.text)}
                          title="Copy message"
                        >
                          {copiedId === m.id ? '✓ Copied' : '⧉ Copy'}
                        </button>
                      )}
                    </div>

                    <div className="msg-body">
                      {m.role === 'agent' ? <FormattedAnswer text={m.text} /> : m.text}
                    </div>

                    {m.meta && (
                      <details className="evidence-accordion">
                        <summary>View Evidence</summary>
                        <div className="evidence-content">
                          {m.meta.tools_called && m.meta.tools_called.length > 0 && (
                            <div className="evidence-row">
                              <span className="evidence-key">Tools Used:</span>
                              <span className="evidence-val">{m.meta.tools_called.join(', ')}</span>
                            </div>
                          )}
                          <div className="evidence-row">
                            <span className="evidence-key">Grounding:</span>
                            <span
                              className="evidence-val"
                              style={{ color: m.meta.grounded ? 'var(--obs)' : 'var(--danger)' }}
                            >
                              {m.meta.grounded ? '✓ Grounded (Verified by Safety Engine)' : 'Unverified'}
                            </span>
                          </div>
                          {answeredBy(m.meta) && (
                            <div className="evidence-row">
                              <span className="evidence-key">Model:</span>
                              <span className="evidence-val">{answeredBy(m.meta)}</span>
                            </div>
                          )}
                          {m.meta.provenance && m.meta.provenance.length > 0 && (
                            <div className="evidence-row">
                              <span className="evidence-key">Sources:</span>
                              <span className="evidence-val">
                                {m.meta.provenance.map((p) => p.source_id).join('; ')}
                              </span>
                            </div>
                          )}
                        </div>
                      </details>
                    )}
                  </div>
                ))}

                {isAsking && (
                  <div className="chat-loading">
                    <span className="brand-icon-pulse" style={{ width: 8, height: 8 }} />
                    <span>Consulting SAT-AI satellite intelligence tools</span>
                    <div className="loading-dots">
                      <span className="loading-dot" />
                      <span className="loading-dot" />
                      <span className="loading-dot" />
                    </div>
                  </div>
                )}

                <div ref={chatBottomRef} />
              </div>

              <form
                className="chat-input-bar"
                onSubmit={(e) => {
                  e.preventDefault();
                  handleSendQuestion(inputQuestion);
                }}
              >
                <input
                  ref={inputRef}
                  type="text"
                  className="chat-input-field"
                  placeholder={`Ask about ${place.name} flood extent, crop damage, terrain, or satellite observations...`}
                  value={inputQuestion}
                  onChange={(e) => setInputQuestion(e.target.value)}
                  disabled={isAsking}
                />
                <button
                  type="submit"
                  className="chat-send-btn"
                  disabled={isAsking || !inputQuestion.trim()}
                >
                  <span>Ask</span>
                  <span>↵</span>
                </button>
              </form>
            </div>
          </div>

          {/* ====================================================================
              15. RESEARCH / METHODOLOGY (Clean Teaser Card)
              ==================================================================== */}
          <div id="research" style={{ marginTop: '56px' }}>
            <div className="research-teaser-card">
              <div className="section-tag">Scientific Methodology &amp; Peer Evaluation</div>
              <h2 className="section-title" style={{ marginTop: '8px' }}>
                Research &amp; <span className="gradient-text">Methodology</span>
              </h2>
              <p className="section-lead">
                SAT-AI was built as an academic AI/ML research project combining Sentinel-1 microwave radar
                and U-Net deep learning segmentation with grounded conversational intelligence.
              </p>

              <div className="research-teaser-grid">
                <div className="research-teaser-item">
                  <h4>📊 Leave-One-Region-Out Validation</h4>
                  <p>
                    Evaluated on a geographically held-out India test region (68 chips) strictly withheld during training.
                    Outperforms classical Otsu thresholding by +0.1476 IoU with 0.6868 F1 score.
                  </p>
                </div>
                <div className="research-teaser-item">
                  <h4>🧠 Dual-Polarization SAR U-Net</h4>
                  <p>
                    7.76M parameter deep learning model processing Sentinel-1 VV, VH, and VV/VH ratio at 10m spatial resolution
                    with radiometric calibration and Range-Doppler Terrain Correction.
                  </p>
                </div>
                <div className="research-teaser-item">
                  <h4>🔍 Explainable AI &amp; Distribution Gating</h4>
                  <p>
                    Pre-inference Wasserstein distribution checks guard against domain shift on unseen satellite acquisitions,
                    with Integrated Gradients pixel attribution for interpretability.
                  </p>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
                <span style={{ fontSize: '13px', color: 'var(--fg-dim)' }}>
                  View complete training splits, loss curves, confusion matrices, and XAI attribution maps.
                </span>
                <Link href="/research" className="btn-primary">
                  <span>View Full Research &amp; Methodology Page</span>
                  <span>→</span>
                </Link>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ====================================================================
          16. FOOTER
          ==================================================================== */}
      <footer className="site-footer">
        <div className="site-container footer-container">
          <div className="footer-left">
            <div className="footer-brand">SAT-AI</div>
            <div className="footer-tag">Understand Any Place From Space.</div>
            <div className="footer-disclaimer">
              Student research prototype — not an official warning system. Official alerts are issued by BSDMA, CWC, and IMD.
            </div>
          </div>

          <ul className="footer-links">
            <li>
              <a
                href="#explore"
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection('explore');
                }}
                className="footer-link"
              >
                Explore
              </a>
            </li>
            <li>
              <a
                href="#disasters"
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection('disasters');
                }}
                className="footer-link"
              >
                Disasters
              </a>
            </li>
            <li>
              <a
                href="#satellite-insights"
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection('satellite-insights');
                }}
                className="footer-link"
              >
                Satellite Insights
              </a>
            </li>
            <li>
              <Link href="/research" className="footer-link">
                Research &amp; Methodology
              </Link>
            </li>
            <li>
              <a
                href="https://github.com/Sahil27-cs/SAT-AI"
                target="_blank"
                rel="noreferrer"
                className="footer-link"
              >
                GitHub
              </a>
            </li>
          </ul>
        </div>
      </footer>
    </>
  );
}
