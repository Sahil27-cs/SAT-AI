'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Image from 'next/image';
import { askAgent, type ChatReply, answeredBy } from '@/lib/api';
import { FormattedAnswer } from '@/components/FormattedAnswer';
import HERO_CHIP from '@/lib/hero-chip.generated.json';

const STARTER_QUESTIONS = [
  'What model did you train?',
  'How much data was used to train it?',
  'What is the India test IoU?',
  'How does U-Net detect flood water?',
  'What are VV, VH and VV/VH ratio?',
  'What do the XAI results mean?',
  "Why can't SAT-AI predict tomorrow's flood?",
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
                  href="#model"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('model');
                  }}
                >
                  Model
                </a>
              </li>
              <li>
                <a
                  href="#results"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('results');
                  }}
                >
                  Results
                </a>
              </li>
              <li>
                <a
                  href="#xai"
                  className="nav-link"
                  onClick={(e) => {
                    e.preventDefault();
                    scrollToSection('xai');
                  }}
                >
                  Explainability
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
                  AI Assistant
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
        {/* 4. HERO SECTION */}
        <section id="hero" className="hero-section">
          <div className="hero-grid">
            <div className="hero-content">
              <div className="hero-eyebrow">
                <span>✦</span>
                <span>AI-POWERED SATELLITE ANALYSIS</span>
              </div>
              <h1 className="hero-title">
                Detecting Floods<br />From Space.
              </h1>
              <p className="hero-desc">
                SAT-AI uses Sentinel-1 SAR imagery and deep learning to segment flood-water
                extent and explain model predictions.
              </p>
              <div className="hero-secondary-line">
                Sentinel-1 SAR • U-Net • Explainable AI • Grounded AI Assistant
              </div>
              <div className="hero-actions">
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => scrollToSection('model')}
                >
                  <span>Explore the Model</span>
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
              {/* Workstation Header */}
              <div className="workstation-header">
                <div className="workstation-dots">
                  <span className="workstation-dot active" title="Active Checkpoint" />
                  <span className="workstation-dot" />
                  <span className="workstation-dot" />
                </div>
                <span>SEN1FLOODS11 // SCENE_0333 // 10M C-BAND SAR</span>
                <span className="hero-visual-badge">EVALUATION CHECKPOINT</span>
              </div>

              {/* Real held-out India test chip visualization: VV backscatter -> Ground truth -> U-Net prediction */}
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
                <span>1. Sentinel-1 VV Backscatter</span>
                <span>2. Hand Label (Ground Truth)</span>
                <span>3. U-Net Flood Extent</span>
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

        {/* 5. HERO METRICS */}
        <section id="metrics" className="metrics-section">
          <div className="metrics-grid">
            <div className="metric-card">
              <div className="metric-val">7.76M</div>
              <div className="metric-label">Model Parameters</div>
            </div>
            <div className="metric-card">
              <div className="metric-val">446</div>
              <div className="metric-label">Hand-Labeled Chips</div>
            </div>
            <div className="metric-card">
              <div className="metric-val">68</div>
              <div className="metric-label">India Test Chips</div>
            </div>
            <div className="metric-card">
              <div className="metric-val">0.523</div>
              <div className="metric-label">India IoU</div>
            </div>
          </div>
          <div className="metric-footer-note">
            Region-disjoint evaluation on an unseen Indian test region
          </div>
        </section>

        {/* 6. HOW IT WORKS */}
        <section id="how-it-works" className="section-padding">
          <div className="section-header centered">
            <div className="section-tag">AI Pipeline Architecture</div>
            <h2 className="section-title">
              From Satellite Signal to <span className="gradient-text">Flood Map</span>
            </h2>
            <p className="section-lead">
              A compact deep-learning pipeline turns Sentinel-1 radar measurements into a pixel-level water mask.
            </p>
          </div>

          <div className="pipeline-grid">
            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">01</span>
                <span className="pipeline-icon">🛰️</span>
              </div>
              <h3 className="pipeline-card-title">SATELLITE INPUT</h3>
              <div className="pipeline-primary">Sentinel-1 SAR</div>
              <div className="pipeline-sub">
                VV + VH dual-polarization radar channels penetrating clouds and rain during active flood events.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">02</span>
                <span className="pipeline-icon">⚡</span>
              </div>
              <h3 className="pipeline-card-title">FEATURE ENGINEERING</h3>
              <div className="pipeline-primary">VV/VH ratio</div>
              <div className="pipeline-sub">
                Calibrated decibel cross-ratio computed at 10 m spatial resolution to highlight specular water reflection.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">03</span>
                <span className="pipeline-icon">🧠</span>
              </div>
              <h3 className="pipeline-card-title">DEEP LEARNING</h3>
              <div className="pipeline-primary">U-Net</div>
              <div className="pipeline-sub">
                7.76M parameter convolutional network with 4 encoder-decoder stages and skip connections for boundary fidelity.
              </div>
            </div>

            <div className="pipeline-card">
              <div className="pipeline-step-header">
                <span className="pipeline-num">04</span>
                <span className="pipeline-icon">🌊</span>
              </div>
              <h3 className="pipeline-card-title">FLOOD SEGMENTATION</h3>
              <div className="pipeline-primary">Probability map → 0.5 threshold → water mask</div>
              <div className="pipeline-sub">
                Continuous sigmoid logit output thresholded at 0.5 into crisp, actionable inundation extents.
              </div>
            </div>
          </div>
        </section>

        {/* 7. MODEL */}
        <section id="model" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Deep Learning Architecture</div>
            <h2 className="section-title">
              The <span className="gradient-text">Model</span>
            </h2>
            <p className="section-lead">
              A U-Net trained from scratch for Sentinel-1 flood-water segmentation.
            </p>
          </div>

          <div className="model-overview-grid">
            <div className="specs-grid">
              <div className="spec-item">
                <div className="spec-key">Architecture</div>
                <div className="spec-value">U-Net (trained from scratch)</div>
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
                <div className="spec-key">Experiment Chips</div>
                <div className="spec-value">446 chips (512×512 px)</div>
              </div>
            </div>

            <div className="model-protocol-card">
              <div className="protocol-header">
                <div className="protocol-title">Training Protocol &amp; Split Breakdown</div>
                <div className="protocol-sub">
                  Leave-one-region-out cross-validation ensuring strict geographic separation:
                </div>
              </div>

              <div className="protocol-chips-grid">
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">333</div>
                  <div className="protocol-chip-label">Training Chips</div>
                </div>
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">30</div>
                  <div className="protocol-chip-label">Validation Chips (Mekong)</div>
                </div>
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">68</div>
                  <div className="protocol-chip-label">India Test Chips</div>
                </div>
                <div className="protocol-chip-box">
                  <div className="protocol-chip-val">15</div>
                  <div className="protocol-chip-label">Reserved Chips (Bolivia)</div>
                </div>
              </div>

              <div className="protocol-distinction-banner">
                <strong>Important Dataset Distinction: </strong>
                Our experiment encompasses <strong>446 total hand-labeled chips</strong> across 11 global flood events.
                Exactly <strong>333 chips</strong> were used to train and optimize the model weights. The remaining chips
                were held out for validation (30 Mekong), unseen regional testing (68 India), and reservation (15 Bolivia).
              </div>
            </div>
          </div>
        </section>

        {/* 8. PERFORMANCE & 9. WHY INDIA HOLDOUT MATTERS */}
        <section id="results" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Empirical Evaluation</div>
            <h2 className="section-title">
              How Well Does It <span className="gradient-text">Perform?</span>
            </h2>
            <p className="section-lead">
              Evaluation on a geographically held-out Indian test region.
            </p>
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
                <span className="comparison-title">U-Net vs Classical Otsu Baseline</span>
                <span className="meta-chip">Held-Out India (68 chips)</span>
              </div>

              <div className="comparison-bars">
                <div className="comp-bar-item">
                  <div className="comp-bar-meta">
                    <strong style={{ color: '#fff' }}>U-Net</strong>
                    <span style={{ color: 'var(--accent)' }}>IoU = 0.523</span>
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
                    <span style={{ color: 'var(--fg-dim)' }}>IoU = 0.375</span>
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
                <span>Improvement: +0.148 IoU over Otsu baseline</span>
              </div>
            </div>

            {/* 9. RESEARCH VALIDATION CALLOUT */}
            <div className="why-matters-card">
              <div>
                <div className="why-matters-title">
                  <span>✦</span>
                  <span>Why the India result matters</span>
                </div>
                <p className="why-matters-text">
                  India was kept geographically separate from training, allowing SAT-AI to test whether
                  the learned representation transfers to an unseen geographic region.
                </p>
              </div>

              <div className="mekong-callout">
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
                  <span className="mekong-val">Mekong validation: 0.868 IoU</span>
                  <span className="mekong-badge">Validation Only</span>
                </div>
                <div>
                  Used strictly for epoch checkpoint selection during training. The held-out India test set (0.523)
                  measures true unseen generalization without data leakage.
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 10. EXPLAINABLE AI */}
        <section id="xai" className="section-padding">
          <div className="section-header">
            <div className="section-tag">Model Interpretability</div>
            <h2 className="section-title">
              Why Did the Model Make <span className="gradient-text">This Prediction?</span>
            </h2>
            <p className="section-lead">
              Model attribution helps us inspect which input features influenced the segmentation.
            </p>
          </div>

          <div className="xai-grid">
            <div className="xai-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20 }}>
                <span style={{ fontWeight: 700, fontSize: 16 }}>Input Feature Attribution</span>
                <span className="meta-chip">Integrated Gradients + Occlusion</span>
              </div>

              <div className="attribution-bar-item">
                <div className="attr-meta">
                  <span className="attr-name">VV/VH Ratio Band</span>
                  <span className="attr-percent">53.4%</span>
                </div>
                <div className="attr-track">
                  <div className="attr-fill primary" style={{ width: '53.4%' }} />
                </div>
                <div className="attr-desc">
                  Largest attribution share, but occlusion disagrees on its sign, and adding it to VV + VH changed test IoU by only 0.0019.
                </div>
              </div>

              <div className="attribution-bar-item">
                <div className="attr-meta">
                  <span className="attr-name">VV Polarisation</span>
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
                  <span className="attr-name">VH Polarisation</span>
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
                <h3 className="xai-info-title">Feature Contribution Analysis</h3>
                <p className="xai-info-text">
                  The VV/VH ratio takes over half of the attribution (53.4%), but attribution is not necessity. In the modality ablation the ratio band alone scored IoU 0.3686, VV alone 0.5279, and all three bands together 0.523: the flood signal lives in absolute backscatter.
                </p>
                <p className="xai-info-text">
                  Shares are Integrated Gradients over 12 held-out India test chips; occlusion was run as a cross-check.
                </p>
              </div>

              <div className="xai-disclaimer">
                <strong>Scientific Disclaimer: </strong>
                These values describe model attribution, not physical causation.
              </div>
            </div>
          </div>
        </section>

        {/* 11. AI ASSISTANT — MAJOR FEATURE */}
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
                placeholder="Ask about U-Net parameters, India IoU benchmark, VV/VH features, or training data..."
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
