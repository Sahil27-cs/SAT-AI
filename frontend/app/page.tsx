'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Image from 'next/image';
import { askAgent, type ChatReply, answeredBy } from '@/lib/api';
import { FormattedAnswer } from '@/components/FormattedAnswer';
import { BiharImpactSection } from '@/components/BiharImpactSection';
import HERO_CHIP from '@/lib/hero-chip.generated.json';

const STARTER_QUESTIONS = [
  'Which Bihar district is most affected?',
  'How much area is flooded?',
  'How much cropland is affected?',
  'Which areas have the highest flood impact?',
  'What changed after the flood?',
  'Which areas are historically flood-prone?',
  'How reliable is this analysis?',
  'What is the India test IoU?',
  'What model did you train?',
  'What do the XAI results mean?',
];

interface ChatMessage {
  id: string;
  role: 'user' | 'agent' | 'error';
  text: string;
  meta?: ChatReply;
}

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputQuestion, setInputQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToSection = (id: string) => {
    setMobileMenuOpen(false);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

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
      {/* 3. NAVBAR */}
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
              <span className="brand-sub">Satellite AI for Flood Detection</span>
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
                  href="#how-it-works"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('how-it-works');
                  }}
                >
                  How It Works
                </a>
              </li>
              <li>
                <a
                  href="#capabilities"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('capabilities');
                  }}
                >
                  Capabilities
                </a>
              </li>
              <li>
                <a
                  href="#bihar-impact"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('bihar-impact');
                  }}
                >
                  Bihar Impact Map
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
                <a
                  href="#scope"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('scope');
                  }}
                >
                  About
                </a>
              </li>
              <li>
                <button
                  type="button"
                  className="nav-cta"
                  onClick={() => {
                    scrollToSection('bihar-impact');
                  }}
                >
                  <span>Explore Impact</span>
                  <span>✦</span>
                </button>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="site-container">
        {/* 4. HERO SECTION */}
        <section id="hero" className="hero-section">
          <div className="hero-grid">
            <div className="hero-content">
              <div className="hero-eyebrow">
                <span>✦</span>
                <span>SATELLITE REMOTE SENSING &amp; FLOOD IMPACT INTELLIGENCE</span>
              </div>
              <h1 className="hero-title">
                Detecting Floods &amp; Impact<br />From Space.
              </h1>
              <p className="hero-desc">
                SAT-AI combines Sentinel-1 SAR satellite imagery with intelligent geospatial analytics to detect flood water extents, evaluate agricultural crop damage, and assess district-level exposure across Bihar.
              </p>
              <div className="hero-secondary-line">
                Sentinel-1 SAR Radar • Bihar Impact Assessment • Cropland Damage • Grounded AI Copilot
              </div>
              <div className="hero-actions">
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => scrollToSection('bihar-impact')}
                >
                  <span>Explore Bihar Map</span>
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

            <div className="hero-visual">
              {/* Satellite Frame Header */}
              <div className="workstation-header">
                <div className="workstation-dots">
                  <span className="workstation-dot active" title="Active Satellite Sensor" />
                  <span className="workstation-dot" />
                  <span className="workstation-dot" />
                </div>
                <span>SENTINEL-1 SATELLITE RADAR // 10M C-BAND // SURFACE WATER DETECTION</span>
                <span className="hero-visual-badge">SATELLITE RADAR CAPTURE</span>
              </div>

              {/* Satellite Radar Surface Water Detection Visualization */}
              <div className="hero-visual-frame">
                <Image
                  src={HERO_CHIP.image}
                  alt="Sentinel-1 radar surface water detection over North Bihar flood plains"
                  width={1568}
                  height={512}
                  className="hero-img"
                  priority
                />
              </div>
              <div className="hero-panel-labels">
                <span>1. Raw Radar Backscatter (SAR)</span>
                <span>2. Reference Inundation Layer</span>
                <span>3. Detected Flood Extent</span>
              </div>
              <div className="hero-visual-bar">
                <span className="hero-visual-badge">North Bihar Flood Plain</span>
                <span>
                  Cloud-penetrating 10m radar imagery · Autonomous surface water delineation · Equal-area metric projection
                </span>
              </div>
              <div className="hero-legend">
                <span><i style={{ background: '#38bdf8' }} />Inundated Flood Water</span>
                <span><i style={{ background: '#f59e0b' }} />Surface Water Spread</span>
                <span><i style={{ background: '#f43f5e' }} />Saturated Soil / Moat</span>
                <span><i style={{ background: '#334155' }} />Dry Land &amp; Settlements</span>
              </div>
            </div>
          </div>
        </section>

        {/* 5. PLATFORM METRICS */}
        <section id="metrics" className="metrics-section">
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-val">10m</div>
              <div className="metric-label">Spatial Resolution</div>
            </div>
            <div className="metric-card">
              <div className="metric-val">38</div>
              <div className="metric-label">Bihar Districts Analyzed</div>
            </div>
            <div className="metric-card">
              <div className="metric-val">3-Class</div>
              <div className="metric-label">Cropland Damage Mapping</div>
            </div>
            <div className="metric-card">
              <div className="metric-val">100%</div>
              <div className="metric-label">Evidence-Based Grounding</div>
            </div>
          </div>
          <div className="metric-footer-note">
            High-precision satellite remote sensing and deterministic geospatial impact intelligence
          </div>
        </section>

        {/* 6. HOW IT WORKS */}
        <section id="how-it-works" className="section-padding">
          <div className="section-header centered">
            <div className="section-tag">End-to-End Workflow</div>
            <h2 className="section-title">
              From Satellite Radar to <span className="gradient-text">Impact Intelligence</span>
            </h2>
            <p className="section-lead">
              Transforming raw orbital radar measurements into actionable flood extent and damage analytics.
            </p>
          </div>

          <div className="pipeline-grid">
            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">01</span>
                <span className="pipeline-icon">🛰️</span>
              </div>
              <h3 className="pipeline-card-title">ORBITAL OBSERVATION</h3>
              <div className="pipeline-primary">Sentinel-1 Radar</div>
              <div className="pipeline-sub">
                C-band Synthetic Aperture Radar penetrates cloud decks and monsoon storms to capture surface reflections day or night.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">02</span>
                <span className="pipeline-icon">🌊</span>
              </div>
              <h3 className="pipeline-card-title">INUNDATION EXTRACTION</h3>
              <div className="pipeline-primary">Surface Water Spread</div>
              <div className="pipeline-sub">
                Autonomous radar backscatter analysis maps open water footprints at 10-meter equal-area resolution.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">03</span>
                <span className="pipeline-icon">🌾</span>
              </div>
              <h3 className="pipeline-card-title">AGRICULTURAL DAMAGE</h3>
              <div className="pipeline-primary">Cropland Assessment</div>
              <div className="pipeline-sub">
                Multi-spectral pre/post optical difference analysis tracks unaffected, partially damaged, and destroyed crops.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">04</span>
                <span className="pipeline-icon">🏛️</span>
              </div>
              <h3 className="pipeline-card-title">DISTRICT IMPACT & COPILOT</h3>
              <div className="pipeline-primary">Exposure Analytics</div>
              <div className="pipeline-sub">
                Deterministic GIS overlays evaluate affected villages, roads, and buildings, queryable via our grounded AI assistant.
              </div>
            </div>
          </div>
        </section>

        {/* 7. CORE PLATFORM CAPABILITIES */}
        <section id="capabilities" className="section-padding">
          <div className="section-header">
            <div className="section-tag">System Capabilities</div>
            <h2 className="section-title">
              Why Satellite Remote Sensing <span className="gradient-text">Matters</span>
            </h2>
            <p className="section-lead">
              Purpose-built capabilities designed specifically for monsoon flood challenges in Bihar.
            </p>
          </div>

          <div className="model-overview-grid">
            <div className="specs-grid">
              <div className="spec-item">
                <div className="spec-key">Sensor Modality</div>
                <div className="spec-value">Synthetic Aperture Radar (SAR) + Optical</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Weather Resilience</div>
                <div className="spec-value">100% cloud-penetrating radar capability</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Projection Standard</div>
                <div className="spec-value">UTM Zone 45N (EPSG:32645) Metric Equal-Area</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Geographic Scope</div>
                <div className="spec-value">All 38 Districts of Bihar, India</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Crop Damage Classes</div>
                <div className="spec-value">No Damage (0), Partial (1), Full Damage (2)</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Severity Index</div>
                <div className="spec-value">SAT-AI Multi-Criteria Impact Index</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Historical Hazard</div>
                <div className="spec-value">NRSC 22-Year Flood Zonation Integration</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Safety Standard</div>
                <div className="spec-value">Strict Zero-Hallucination Policy</div>
              </div>
            </div>

            <div className="model-protocol-card">
              <div className="protocol-header">
                <div className="protocol-title">Why Radar Sees What Optical Cameras Cannot</div>
                <div className="protocol-sub">
                  Optical satellites are blinded by monsoon storm clouds during active disaster peaks:
                </div>
              </div>

              <div className="protocol-chips-grid">
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">Day &amp; Night</div>
                  <div className="protocol-chip-label">Active Microwave Pulse</div>
                </div>
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">Cloud-Free</div>
                  <div className="protocol-chip-label">Unblocked by Monsoons</div>
                </div>
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">Mirror Effect</div>
                  <div className="protocol-chip-label">Specular Water Reflection</div>
                </div>
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">10-Meter</div>
                  <div className="protocol-chip-label">Field-Level Precision</div>
                </div>
              </div>

              <div className="protocol-distinction-banner">
                <strong>Physics of Satellite Water Detection: </strong>
                Calm flood water behaves like an electromagnetic mirror for satellite radar beams, bouncing the microwave energy away into space. This produces the characteristic dark signature on radar imagery, allowing automated detection of inundated landscapes.
              </div>
            </div>
          </div>
        </section>


        {/* 11. BIHAR FLOOD IMPACT ASSESSMENT & INTERACTIVE MAP */}
        <BiharImpactSection />

        {/* 12. AI ASSISTANT — MAJOR FEATURE */}
        <section id="assistant" className="assistant-section">
          <div className="section-header centered">
            <div className="section-tag">Interactive Research Copilot</div>
            <h2 className="section-title">
              Ask <span className="gradient-text">SAT-AI</span>
            </h2>
            <p className="section-lead">
              Explore the model, dataset, results and scientific limitations through a grounded AI assistant.
            </p>
          </div>

          <div className="assistant-card">
            <div className="assistant-header">
              <div className="assistant-header-left">
                <span className="assistant-status-beacon" />
                <span className="assistant-header-title">SAT-AI Research Assistant</span>
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

            <div className="starter-prompts">
              <div className="starter-prompts-label">Suggested Research Inquiries:</div>
              <div className="prompt-chips-grid">
                {STARTER_QUESTIONS.map((q) => (
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

            <div className="chat-window">
              {messages.length === 0 && (
                <div className="chat-empty-state">
                  <div className="chat-empty-icon">🛰️</div>
                  <div className="chat-empty-title">Flood Intelligence Copilot Ready</div>
                  <div className="chat-empty-text">
                    Ask any question above about Bihar flood inundation, affected districts,
                    cropland damage, infrastructure exposure, or satellite observation dates.
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
                    {m.role === 'agent' ? (
                      <FormattedAnswer text={m.text} />
                    ) : (
                      m.text
                    )}
                  </div>

                  {m.meta && (
                    <details className="evidence-accordion">
                      <summary>View Evidence</summary>
                      <div className="evidence-content">
                        {m.meta.tools_called && m.meta.tools_called.length > 0 && (
                          <div className="evidence-row">
                            <span className="evidence-key">Tools Used:</span>
                            <span className="evidence-val">
                              {m.meta.tools_called.join(', ')}
                            </span>
                          </div>
                        )}
                        <div className="evidence-row">
                          <span className="evidence-key">Grounding:</span>
                          <span className="evidence-val" style={{ color: m.meta.grounded ? 'var(--obs)' : 'var(--danger)' }}>
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
                  <span>Consulting SAT-AI flood intelligence tools</span>
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
                placeholder="Ask about Bihar flood inundation, affected districts, crop damage, or satellite observations..."
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
        </section>

        {/* 17. RESEARCH SCOPE */}
        <section id="scope" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Scientific Integrity</div>
            <h2 className="section-title">
              Research <span className="gradient-text">Scope</span>
            </h2>
            <p className="section-lead">
              Clear methodological boundaries defining the research prototype and its evaluation scope.
            </p>
          </div>

          <div className="scope-grid">
            <div className="scope-card">
              <span className="scope-bullet">•</span>
              <div className="scope-text">
                <strong>Sentinel-1 revisit time</strong> means this is not continuous real-time flood forecasting.
              </div>
            </div>

            <div className="scope-card">
              <span className="scope-bullet">•</span>
              <div className="scope-text">
                <strong>New satellite scenes</strong> without ground-truth labels cannot receive IoU/F1 evaluation.
              </div>
            </div>

            <div className="scope-card">
              <span className="scope-bullet">•</span>
              <div className="scope-text">
                <strong>SAR backscatter</strong> can be challenging in dense urban environments due to building radar shadow and double-bounce effects.
              </div>
            </div>

            <div className="scope-card">
              <span className="scope-bullet">•</span>
              <div className="scope-text">
                <strong>SAT-AI is a student research prototype</strong>, not an official government warning system.
              </div>
            </div>
          </div>

          <div className="scope-notes-box">
            <div>
              SAT-AI is designed as a broader multi-hazard research prototype, while the validated ML pipeline presented here focuses on flood-water segmentation.
            </div>
            <div>
              SAT-AI also contains prototype risk-analysis components, but regional risk maps are withheld when validated hazard, exposure and vulnerability data are unavailable.
            </div>
          </div>
        </section>
      </main>

      {/* 18. FOOTER */}
      <footer className="site-footer">
        <div className="site-container footer-container">
          <div className="footer-left">
            <div className="footer-brand">SAT-AI</div>
            <div className="footer-tag">AI-powered flood detection from satellite imagery.</div>
            <div className="footer-disclaimer">
              Student research prototype — not an official warning system.
            </div>
          </div>

          <ul className="footer-links">
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
            <li>
              <a
                href="#assistant"
                onClick={(e) => {
                  e.preventDefault();
                  scrollToSection('assistant');
                  inputRef.current?.focus();
                }}
                className="footer-link"
              >
                AI Assistant
              </a>
            </li>
          </ul>
        </div>
      </footer>
    </>
  );
}
