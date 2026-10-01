'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Image from 'next/image';
import { askAgent, type ChatReply, answeredBy } from '@/lib/api';
import { FormattedAnswer } from '@/components/FormattedAnswer';
import HERO_CHIP from '@/lib/hero-chip.generated.json';

const STARTER_QUESTIONS = [
  'What model do you use for flood detection?',
  'on how much dataset u have trained model',
  'What is the India test IoU?',
  'How does the U-Net detect flood water?',
  'What are the VV, VH and VV/VH ratio features?',
  'What happened to the 74.8 km² Nepal result?',
  "Why can't SAT-AI predict whether Mumbai will flood tomorrow?",
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
    // The Clipboard API is missing outside secure contexts and can be denied;
    // show "copied" only when the write actually succeeded.
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
      {/* 19. NAVIGATION */}
      <header className="site-header">
        <div className="site-container nav-container">
          <a href="#hero" className="brand-link" onClick={(e) => { e.preventDefault(); scrollToSection('hero'); }}>
            <div className="brand-icon">
              <span className="brand-icon-pulse" />
            </div>
            <div className="brand-text">
              <span className="brand-title">SAT-AI</span>
              <span className="brand-sub">AI/ML RESEARCH</span>
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
                  href="#model"
                  className="nav-link"
                  onClick={(e) => { e.preventDefault(); scrollToSection('model'); }}
                >
                  Model
                </a>
              </li>
              <li>
                <a
                  href="#results"
                  className="nav-link"
                  onClick={(e) => { e.preventDefault(); scrollToSection('results'); }}
                >
                  Results
                </a>
              </li>
              <li>
                <a
                  href="#xai"
                  className="nav-link"
                  onClick={(e) => { e.preventDefault(); scrollToSection('xai'); }}
                >
                  Explainability
                </a>
              </li>
              <li>
                <a
                  href="#assistant"
                  className="nav-link"
                  onClick={(e) => { e.preventDefault(); scrollToSection('assistant'); }}
                >
                  AI Assistant
                </a>
              </li>
              <li>
                <a
                  href="#about"
                  className="nav-link"
                  onClick={(e) => { e.preventDefault(); scrollToSection('about'); }}
                >
                  About
                </a>
              </li>
              <li>
                <button
                  type="button"
                  className="nav-cta"
                  onClick={() => {
                    scrollToSection('assistant');
                    inputRef.current?.focus();
                  }}
                >
                  <span>Ask SAT-AI</span>
                  <span>✦</span>
                </button>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="site-container">
        {/* 5. HERO SECTION */}
        <section id="hero" className="hero-section">
          <div className="hero-grid">
            <div className="hero-content">
              <div className="hero-badge">
                <span>✦</span>
                <span>STUDENT AI/ML RESEARCH PROJECT · SEN1FLOODS11</span>
              </div>
              <h1 className="hero-title">SAT-AI</h1>
              <h2 className="hero-subtitle">
                AI-Powered Flood Detection from Satellite Imagery
              </h2>
              <p className="hero-desc">
                SAT-AI uses Sentinel-1 SAR imagery and deep learning to identify flood-water
                extent, evaluate model performance on unseen geographic regions, and provide
                grounded explanations through an AI assistant.
              </p>
              <div className="hero-actions">
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => {
                    scrollToSection('assistant');
                    inputRef.current?.focus();
                  }}
                >
                  <span>Try AI Assistant</span>
                  <span>→</span>
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => scrollToSection('results')}
                >
                  <span>View Model Results</span>
                </button>
              </div>
              <div className="hero-meta-tags">
                <span className="meta-chip">Sentinel-1 SAR</span>
                <span className="meta-chip">U-Net 7.76M</span>
                <span className="meta-chip">Region-Disjoint LORO</span>
                <span className="meta-chip">Grounded Copilot</span>
              </div>
            </div>

            <div className="hero-visual">
              {/* Workstation Chrome Header */}
              <div className="workstation-header">
                <div className="workstation-dots">
                  <span className="workstation-dot active" title="Active Checkpoint" />
                  <span className="workstation-dot" />
                  <span className="workstation-dot" />
                </div>
                <span>SEN1FLOODS11 // SCENE_0333 // 10M C-BAND SAR</span>
                <span className="hero-visual-badge">EVALUATION CHECKPOINT</span>
              </div>

              {/* A real held-out India chip, rendered by scripts/make_hero_figure.py
                  from the trained checkpoint. Not the best chip: the one whose own
                  IoU is closest to the pooled test score. */}
              <div className="hero-visual-frame">
                <Image
                  src={HERO_CHIP.image}
                  alt={`Held-out India test chip ${HERO_CHIP.chip}: Sentinel-1 VV backscatter, the hand-labelled ground truth, and the U-Net prediction compared against it`}
                  width={1568}
                  height={512}
                  className="hero-img"
                  priority
                />
              </div>
              <div className="hero-panel-labels">
                <span>VV backscatter (dB)</span>
                <span>Hand label (ground truth)</span>
                <span>U-Net prediction</span>
              </div>
              <div className="hero-visual-bar">
                <span className="hero-visual-badge">Held-out India chip {HERO_CHIP.chip}</span>
                <span>
                  IoU {HERO_CHIP.chip_iou} on this chip · pooled test IoU {HERO_CHIP.pooled_test_iou} ·
                  per-chip median {HERO_CHIP.per_chip_median_iou.toFixed(2)}
                </span>
              </div>
              <div className="hero-legend">
                <span><i style={{ background: '#38bdf8' }} />water, agreed</span>
                <span><i style={{ background: '#f59e0b' }} />water missed</span>
                <span><i style={{ background: '#f43f5e' }} />false water</span>
                <span><i style={{ background: '#334155' }} />not labelled</span>
              </div>
            </div>
          </div>
        </section>

        {/* 6. SIMPLE PROJECT STATISTICS */}
        <section id="metrics" className="metrics-section">
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-tag">01 // ARCHITECTURE</span>
                <span className="metric-badge">PyTorch 2.x</span>
              </div>
              <div className="metric-val">7.76M</div>
              <div className="metric-label">Model Parameters</div>
              <div className="metric-sub">Custom U-Net trained from scratch with 4 encoder-decoder stages</div>
            </div>
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-tag">02 // DATASET</span>
                <span className="metric-badge">11 Flood Events</span>
              </div>
              <div className="metric-val">446</div>
              <div className="metric-label">Training/Evaluation Chips</div>
              <div className="metric-sub">Sen1Floods11 v1.1 hand-labeled tiles across 11 global flood events</div>
            </div>
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-tag">03 // GENERALIZATION</span>
                <span className="metric-badge">Held-Out Test</span>
              </div>
              <div className="metric-val">68</div>
              <div className="metric-label">Held-out India Test Chips</div>
              <div className="metric-sub">Geographically isolated test region unseen during training</div>
            </div>
            <div className="metric-card">
              <div className="metric-header">
                <span className="metric-tag">04 // BENCHMARK</span>
                <span className="metric-badge">+39.5% Rel. Gain</span>
              </div>
              <div className="metric-val">0.523</div>
              <div className="metric-label">India Test IoU</div>
              <div className="metric-sub">+0.148 improvement over classical Otsu baseline thresholding</div>
            </div>
          </div>
        </section>

        {/* 7. HOW IT WORKS */}
        <section id="how-it-works" className="section-padding">
          <div className="section-header centered">
            <div className="section-tag">AI Pipeline Architecture</div>
            <h2 className="section-title">
              How It <span className="gradient-text">Works</span>
            </h2>
            <p className="section-lead">
              A 4-step workflow converting raw all-weather satellite radar backscatter
              into high-resolution inundation segmentation masks.
            </p>
          </div>

          <div className="pipeline-grid">
            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">01</span>
                <span className="pipeline-icon">🛰️</span>
              </div>
              <h3 className="pipeline-card-title">Satellite Input</h3>
              <div className="pipeline-primary">Sentinel-1 SAR</div>
              <div className="pipeline-sub">
                Dual-polarization radar returns (VV + VH) penetrating clouds and rain during active flood events.
              </div>
              <div className="pipeline-spec-tag">10m C-Band SAR Dual-Pol</div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">02</span>
                <span className="pipeline-icon">⚡</span>
              </div>
              <h3 className="pipeline-card-title">Feature Engineering</h3>
              <div className="pipeline-primary">VV/VH ratio</div>
              <div className="pipeline-sub">
                Calibrated decibel conversion and cross-polarization ratio generation at 10 m spatial resolution.
              </div>
              <div className="pipeline-spec-tag">Calibrated dB &amp; Cross-Ratio</div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">03</span>
                <span className="pipeline-icon">🧠</span>
              </div>
              <h3 className="pipeline-card-title">AI Segmentation</h3>
              <div className="pipeline-primary">U-Net (7.76M)</div>
              <div className="pipeline-sub">
                Deep convolutional network with skip connections, trained with joint 0.5 BCE + 0.5 Dice loss.
              </div>
              <div className="pipeline-spec-tag">4 Encoder-Decoder Stages</div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">04</span>
                <span className="pipeline-icon">🌊</span>
              </div>
              <h3 className="pipeline-card-title">Flood Mask</h3>
              <div className="pipeline-primary">Water Probability</div>
              <div className="pipeline-sub">
                Continuous pixel-level sigmoid probabilities classified into clean flood extents at threshold = 0.5.
              </div>
              <div className="pipeline-spec-tag">Water Probability (≥ 0.5)</div>
            </div>
          </div>
        </section>

        {/* 8. MODEL SECTION */}
        <section id="model" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Deep Learning Architecture</div>
            <h2 className="section-title">
              Inside the <span className="gradient-text">Flood Detection Model</span>
            </h2>
            <p className="section-lead">
              SAT-AI uses a U-Net trained from scratch to segment water from Sentinel-1 SAR imagery.
            </p>
          </div>

          <div className="model-overview-grid">
            <div className="specs-grid">
              <div className="spec-item">
                <div className="spec-key">Architecture</div>
                <div className="spec-value">U-Net (from scratch)</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Parameters</div>
                <div className="spec-value">7.76M (7,763,041)</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Input Bands</div>
                <div className="spec-value">VV + VH + VV/VH ratio</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Loss Function</div>
                <div className="spec-value">0.5 BCE + 0.5 Dice</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Output</div>
                <div className="spec-value">Pixel-level water probability</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Decision Threshold</div>
                <div className="spec-value">0.5 (sigmoid probability)</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Dataset</div>
                <div className="spec-value">Sen1Floods11 v1.1 Hand-Labeled</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Dataset Size Used</div>
                <div className="spec-value">446 chips (512×512 px)</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Flood Events</div>
                <div className="spec-value">11 global events</div>
              </div>
              <div className="spec-item">
                <div className="spec-key">Test Protocol</div>
                <div className="spec-value">Region-disjoint India holdout</div>
              </div>
            </div>

            <div className="model-architecture-card">
              <div>
                <div className="arch-header">
                  <span className="arch-title">U-Net Topology</span>
                  <span className="meta-chip">PyTorch 2.x</span>
                </div>
                <p style={{ fontSize: 13, color: 'var(--fg-dim)', marginBottom: 12 }}>
                  The 4-level encoder progressively extracts multi-scale spatial representations, while
                  skip connections preserve sharp boundaries for precise waterline localization.
                </p>
                <div className="arch-diagram">
                  <div>Input: (3 × 512 × 512) [VV, VH, VV/VH]</div>
                  <div>│</div>
                  <div>▼ Encoder Stage 1: Conv 32 <span className="arch-skip">────────────┐ (skip)</span></div>
                  <div>▼ Encoder Stage 2: Conv 64 <span className="arch-skip">──────────┐ │</span></div>
                  <div>▼ Encoder Stage 3: Conv 128 <span className="arch-skip">────────┐ │ │</span></div>
                  <div>▼ Encoder Stage 4: Conv 256 <span className="arch-skip">──────┐ │ │ │</span></div>
                  <div>▼ Bottleneck: <span className="arch-node">Conv 512 + Dropout</span> │ │ │ │</div>
                  <div>▲ Decoder Stage 4: Conv 256 <span className="arch-skip">◀─────┘ │ │ │</span></div>
                  <div>▲ Decoder Stage 3: Conv 128 <span className="arch-skip">◀───────┘ │ │</span></div>
                  <div>▲ Decoder Stage 2: Conv 64 <span className="arch-skip">◀─────────┘ │</span></div>
                  <div>▲ Decoder Stage 1: Conv 32 <span className="arch-skip">◀───────────┘</span></div>
                  <div>│</div>
                  <div>Output: (1 × 512 × 512) [Water Logit → Sigmoid]</div>
                </div>
              </div>
              <div style={{ fontSize: 12, color: 'var(--fg-faint)', fontStyle: 'italic', marginTop: 10 }}>
                Trained with AdamW (lr = 3e-4, weight decay 1e-4, cosine schedule), batch size 8, 32 epochs; the checkpoint was selected on Mekong validation IoU.
              </div>
            </div>
          </div>
        </section>

        {/* 9. MODEL PERFORMANCE & 10. WHY THIS MATTERS */}
        <section id="results" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Empirical Evaluation</div>
            <h2 className="section-title">
              How Well Does the <span className="gradient-text">Model Perform?</span>
            </h2>
            <p className="section-lead">
              Rigorous quantitative benchmark on a strictly held-out geographic region.
            </p>
          </div>

          <div style={{ marginBottom: 16 }}>
            <span className="hero-visual-badge" style={{ fontSize: 12 }}>
              India — geographically held-out test set (68 chips)
            </span>
          </div>

          <div className="results-metric-grid">
            <div className="result-card highlight">
              <div className="result-metric-val">0.523</div>
              <div className="result-metric-name">IoU Score</div>
            </div>
            <div className="result-card">
              <div className="result-metric-val">0.687</div>
              <div className="result-metric-name">F1 Score</div>
            </div>
            <div className="result-card">
              <div className="result-metric-val">0.751</div>
              <div className="result-metric-name">Precision</div>
            </div>
            <div className="result-card">
              <div className="result-metric-val">0.633</div>
              <div className="result-metric-name">Recall</div>
            </div>
          </div>

          <div className="comparison-container">
            <div className="comparison-card">
              <div className="comparison-header">
                <span className="comparison-title">Benchmark Comparison: India Test Set</span>
                <span className="meta-chip">Region-Disjoint</span>
              </div>

              <div className="comparison-bars">
                <div className="comp-bar-item">
                  <div className="comp-bar-meta">
                    <strong style={{ color: '#fff' }}>SAT-AI U-Net</strong>
                    <span style={{ color: 'var(--accent)' }}>0.523 IoU</span>
                  </div>
                  <div className="comp-bar-track">
                    <div className="comp-bar-fill model" style={{ width: '52.3%' }}>
                      0.523
                    </div>
                  </div>
                </div>

                <div className="comp-bar-item">
                  <div className="comp-bar-meta">
                    <span style={{ color: 'var(--fg-dim)' }}>Otsu Baseline</span>
                    <span style={{ color: 'var(--fg-dim)' }}>0.375 IoU</span>
                  </div>
                  <div className="comp-bar-track">
                    <div className="comp-bar-fill baseline" style={{ width: '37.5%' }}>
                      0.375
                    </div>
                  </div>
                </div>
              </div>

              <div className="improvement-badge">
                <span>▲</span>
                <span>Improvement: +0.148 IoU (+39.3% relative to the Otsu baseline)</span>
              </div>
            </div>

            <div className="why-matters-card">
              <div>
                <div className="why-matters-title">
                  <span>✦</span>
                  <span>Why This Matters</span>
                </div>
                <p className="why-matters-text">
                  Unlike a random image split, the India region was kept geographically separate
                  from training. This tests whether the model can generalize to an unseen
                  geographic region.
                </p>
              </div>

              <div className="mekong-callout">
                <div style={{ marginBottom: 4 }}>
                  <span className="mekong-val">Mekong validation IoU: 0.868</span>
                </div>
                <div>
                  Fold validation score during model selection. The India test set (0.523) was held
                  out completely to evaluate true cross-geographic generalization without data leakage.
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 11. XAI SECTION */}
        <section id="xai" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Model Interpretability</div>
            <h2 className="section-title">
              Why Did the Model Make Its <span className="gradient-text">Prediction?</span>
            </h2>
            <p className="section-lead">
              Integrated Gradients over 12 held-out India chips shows which radar inputs the network
              relied on. Occlusion was run as a cross-check and disagrees on one band.
            </p>
          </div>

          <div className="xai-grid">
            <div className="xai-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
                <span style={{ fontWeight: 700, fontSize: 16 }}>Input Feature Attribution</span>
                <span className="meta-chip">Integrated Gradients share</span>
              </div>

              <div className="attribution-bar-item">
                <div className="attr-meta">
                  <span className="attr-name">
                    <span>✦</span>
                    <span>VV/VH ratio feature</span>
                  </span>
                  <span className="attr-percent">53.4%</span>
                </div>
                <div className="attr-track">
                  <div className="attr-fill primary" style={{ width: '53.4%' }} />
                </div>
                <div className="attr-desc">
                  Largest share, but occlusion disagrees on its sign, and in the modality ablation adding it to VV + VH changed test IoU by only 0.0019.
                </div>
              </div>

              <div className="attribution-bar-item">
                <div className="attr-meta">
                  <span className="attr-name">
                    <span>✦</span>
                    <span>VV polarization band</span>
                  </span>
                  <span className="attr-percent">29.8%</span>
                </div>
                <div className="attr-track">
                  <div className="attr-fill" style={{ width: '29.8%' }} />
                </div>
                <div className="attr-desc">
                  Co-polarised backscatter. Calm open water is dark in VV.
                </div>
              </div>

              <div className="attribution-bar-item">
                <div className="attr-meta">
                  <span className="attr-name">
                    <span>✦</span>
                    <span>VH polarization band</span>
                  </span>
                  <span className="attr-percent">16.8%</span>
                </div>
                <div className="attr-track">
                  <div className="attr-fill" style={{ width: '16.8%' }} />
                </div>
                <div className="attr-desc">
                  Cross-polarised backscatter. Also low over open water; higher over vegetation.
                </div>
              </div>
            </div>

            <div className="xai-info-card">
              <div>
                <div className="xai-info-title">How to read this</div>
                <p className="xai-info-text">
                  Calm open water reflects the radar pulse away from the satellite, so it appears dark in VV and
                  VH; that is the physics. The shares above are a different thing: how much the trained network&apos;s
                  output moved with each input. A high share does not prove a band is needed. The modality
                  ablation is the test of that, and there VV alone scored as well as all three bands together.
                </p>
              </div>

              <div className="xai-disclaimer">
                <strong>Attribution Note: </strong>
                These values describe model attribution — which input features the model relied on.
                They do not represent physical causation.
              </div>
            </div>
          </div>
        </section>

        {/* 12. AI ASSISTANT — MAJOR FEATURE */}
        <section id="assistant" className="assistant-section">
          <div className="section-header centered">
            <div className="section-tag">Interactive Research Copilot</div>
            <h2 className="section-title">
              Ask <span className="gradient-text">SAT-AI</span>
            </h2>
            <p className="section-lead">
              Ask questions about the model, its results, satellite inputs and flood analysis.
            </p>
          </div>

          <div className="assistant-card">
            <div className="assistant-header">
              <div className="assistant-header-left">
                <span className="assistant-status-beacon" />
                <span className="assistant-header-title">SAT-AI Research Assistant</span>
                <span className="assistant-badge">4-CHECK GROUNDING VALIDATOR</span>
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
                    Ask any question above about the U-Net architecture, held-out India test IoU,
                    SAR polarizations, or policy refusals.
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
                      <summary>View evidence</summary>
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
                placeholder="Ask about U-Net parameters, India IoU benchmark, VV/VH features, or Nepal scene..."
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

        {/* 16. RESEARCH SCOPE & LIMITATIONS */}
        <section id="limitations" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Scientific Integrity</div>
            <h2 className="section-title">
              Research Scope &amp; <span className="gradient-text">Limitations</span>
            </h2>
            <p className="section-lead">
              Clear methodological boundaries defining what the prototype is designed to evaluate.
            </p>
          </div>

          <div className="limitations-grid">
            <div className="limitation-card">
              <div className="limitation-icon">⏱️</div>
              <div className="limitation-text">
                <strong>Sentinel-1 revisit time</strong> means this is not continuous real-time flood forecasting.
              </div>
            </div>

            <div className="limitation-card">
              <div className="limitation-icon">📊</div>
              <div className="limitation-text">
                <strong>New satellite scenes</strong> without ground truth cannot receive an F1/IoU score.
              </div>
            </div>

            <div className="limitation-card">
              <div className="limitation-icon">🏙️</div>
              <div className="limitation-text">
                <strong>Urban areas</strong> can be challenging because SAR backscatter is affected by buildings and double-bounce effects.
              </div>
            </div>

            <div className="limitation-card">
              <div className="limitation-icon">🛡️</div>
              <div className="limitation-text">
                <strong>SAT-AI is a research prototype</strong>, not an official government warning system.
              </div>
            </div>
          </div>
        </section>

        {/* 17. ABOUT SECTION */}
        <section id="about" className="section-padding">
          <div className="about-card">
            <div className="about-header">
              <div className="about-title">Built as an AI/ML Research Project</div>
              <div className="about-subtitle">Vidyalankar Institute of Technology (VIT), Mumbai</div>
            </div>

            <p className="about-lead">
              SAT-AI was built to explore how deep learning and satellite radar remote sensing can automate
              flood-water extent mapping across geographically held-out regions, paired with an evidence-grounded
              conversational agent for transparent interpretability.
            </p>

            <div className="about-pillars">
              <div className="pillar-card">
                <div className="pillar-title">
                  <span>🛰️</span> Sentinel-1 SAR
                </div>
                <div className="pillar-desc">
                  Dual-polarized C-band Synthetic Aperture Radar providing 10m all-weather imaging independent of cloud cover.
                </div>
              </div>

              <div className="pillar-card">
                <div className="pillar-title">
                  <span>🧠</span> U-Net Architecture
                </div>
                <div className="pillar-desc">
                  Custom 7.76M parameter convolutional network trained from scratch on Sen1Floods11 hand-labeled tiles.
                </div>
              </div>

              <div className="pillar-card">
                <div className="pillar-title">
                  <span>🔬</span> Explainable AI &amp; Agent
                </div>
                <div className="pillar-desc">
                  Integrated Gradients and occlusion attribution, and a 4-check grounding validator on every Gemini answer.
                </div>
              </div>
            </div>

            <div className="about-disclaimer-box">
              <p style={{ marginBottom: 6 }}>
                SAT-AI is designed as a broader multi-hazard research prototype, while the experimentally
                validated ML pipeline presented here focuses on flood-water segmentation. Prototype risk-analysis
                components exist in the backend, but regional risk maps are not presented where validated
                hazard/exposure data are unavailable.
              </p>
              <p style={{ color: 'var(--fg-faint)', fontSize: 12 }}>
                Policy notice: SAT-AI performs no earthquake prediction, does not predict future floods, and does not issue official disaster alerts.
              </p>
            </div>
          </div>
        </section>
      </main>

      {/* 20. FOOTER */}
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
                href="https://github.com/Sahil27-cs/SAT-AI/tree/main/docs"
                target="_blank"
                rel="noreferrer"
                className="footer-link"
              >
                Documentation
              </a>
            </li>
          </ul>
        </div>
      </footer>
    </>
  );
}
