'use client';

import { useState } from 'react';
import HERO_CHIP from '@/lib/hero-chip.generated.json';
import Image from 'next/image';

export function ResearchSection() {
  const [activeTab, setActiveTab] = useState<'benchmarks' | 'architecture' | 'interpretability' | 'datasets'>('benchmarks');

  return (
    <section id="research" className="research-methodology-section section-padding">
      <div className="section-header centered">
        <div className="section-tag">Scientific Methodology &amp; Peer Evaluation</div>
        <h2 className="section-title">
          Research &amp; <span className="gradient-text">Methodology</span>
        </h2>
        <p className="section-lead">
          Technical specifications, leave-one-region-out empirical validation, and model interpretability for peer review.
        </p>

        {/* Tab Switcher */}
        <div className="research-tab-bar">
          <button
            type="button"
            className={`research-tab-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmarks')}
          >
            📊 Model Benchmarks
          </button>
          <button
            type="button"
            className={`research-tab-btn ${activeTab === 'architecture' ? 'active' : ''}`}
            onClick={() => setActiveTab('architecture')}
          >
            🧠 U-Net Architecture
          </button>
          <button
            type="button"
            className={`research-tab-btn ${activeTab === 'interpretability' ? 'active' : ''}`}
            onClick={() => setActiveTab('interpretability')}
          >
            🔍 XAI &amp; Attribution
          </button>
          <button
            type="button"
            className={`research-tab-btn ${activeTab === 'datasets' ? 'active' : ''}`}
            onClick={() => setActiveTab('datasets')}
          >
            📁 Dataset Protocols
          </button>
        </div>
      </div>

      <div className="research-card-container">
        {/* TAB 1: EMPIRICAL BENCHMARKS */}
        {activeTab === 'benchmarks' && (
          <div className="research-content-block">
            <div className="research-block-header">
              <h3>Empirical Evaluation on Geographically Held-Out India Test Region</h3>
              <p>
                To evaluate true spatial generalization without data leakage, the India test region (68 chips) was strictly withheld during model training and checkpoint selection.
              </p>
            </div>

            <div className="benchmark-metrics-grid">
              <div className="benchmark-card highlight-cyan">
                <div className="b-val">0.5230</div>
                <div className="b-label">India Test IoU Score</div>
                <div className="b-sub">Intersection-over-Union</div>
              </div>
              <div className="benchmark-card">
                <div className="b-val">0.6868</div>
                <div className="b-label">India Test F1 Score</div>
                <div className="b-sub">Harmonic Mean Precision/Recall</div>
              </div>
              <div className="benchmark-card">
                <div className="b-val">0.7506</div>
                <div className="b-label">Precision</div>
                <div className="b-sub">Water Classification Accuracy</div>
              </div>
              <div className="benchmark-card">
                <div className="b-val">0.6331</div>
                <div className="b-label">Recall</div>
                <div className="b-sub">True Positive Water Recovery</div>
              </div>
            </div>

            {/* Comparison vs Otsu Baseline */}
            <div className="otsu-comparison-box">
              <div className="otsu-header">
                <strong>U-Net vs Classical Otsu Threshold Baseline</strong>
                <span className="otsu-pill">+0.1476 IoU Improvement</span>
              </div>
              <div className="otsu-bar-wrapper">
                <div className="otsu-bar-label">
                  <span>SAT-AI U-Net (Deep Learning)</span>
                  <span className="val cyan">IoU = 0.5230</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill cyan" style={{ width: '52.3%' }}>0.5230</div>
                </div>
              </div>
              <div className="otsu-bar-wrapper">
                <div className="otsu-bar-label">
                  <span>Classical Otsu Intensity Baseline</span>
                  <span className="val dim">IoU = ~0.3754</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill dim" style={{ width: '37.5%' }}>0.3754</div>
                </div>
              </div>
              <div className="otsu-note">
                Classical Otsu thresholding struggles with radar speckle and dielectric surface variations; the deep convolutional U-Net captures spatial context and multi-polarization cross-ratio information.
              </div>
            </div>

            {/* Real Held-Out Chip Visualization */}
            <div className="chip-eval-view">
              <div className="chip-view-title">
                Held-Out India Evaluation Chip (Scene 0333 / Chip {HERO_CHIP.chip})
              </div>
              <div className="chip-img-frame">
                <Image
                  src={HERO_CHIP.image}
                  alt="Held-out India test chip comparison"
                  width={1568}
                  height={512}
                  className="chip-img"
                />
              </div>
              <div className="chip-meta-line">
                <span>Chip IoU: {HERO_CHIP.chip_iou}</span>
                <span>• Pooled Test IoU: {HERO_CHIP.pooled_test_iou}</span>
                <span>• Per-Chip Median: {HERO_CHIP.per_chip_median_iou.toFixed(2)}</span>
                <span>• Sensor: Sentinel-1 C-SAR IW GRD</span>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: ARCHITECTURE SPECIFICATIONS */}
        {activeTab === 'architecture' && (
          <div className="research-content-block">
            <div className="research-block-header">
              <h3>Deep Learning Model Architecture &amp; Hyperparameters</h3>
              <p>
                Fully trained from scratch without pre-trained weights to evaluate pure SAR feature extraction.
              </p>
            </div>

            <div className="arch-specs-table">
              <div className="spec-row">
                <span className="spec-name">Model Family</span>
                <span className="spec-info">Convolutional Encoder-Decoder U-Net with Skip Connections</span>
              </div>
              <div className="spec-row">
                <span className="spec-name">Total Parameters</span>
                <span className="spec-info mono">7,763,041 (~7.76 Million Parameters)</span>
              </div>
              <div className="spec-row">
                <span className="spec-name">Input Tensor Channels</span>
                <span className="spec-info">3 Bands: VV (dB), VH (dB), and VV/VH Cross-Ratio</span>
              </div>
              <div className="spec-row">
                <span className="spec-name">Spatial Tile Dimensions</span>
                <span className="spec-info">512 × 512 pixels at 10-meter ground sample distance</span>
              </div>
              <div className="spec-row">
                <span className="spec-name">Loss Function Formulation</span>
                <span className="spec-info mono">0.50 × Binary Cross-Entropy + 0.50 × Soft Dice Loss</span>
              </div>
              <div className="spec-row">
                <span className="spec-name">Optimization Algorithm</span>
                <span className="spec-info">AdamW (Weight Decay: 1e-4, Initial Learning Rate: 3e-4)</span>
              </div>
              <div className="spec-row">
                <span className="spec-name">Decision Threshold</span>
                <span className="spec-info">0.50 (Sigmoid Water Probability Logits)</span>
              </div>
              <div className="spec-row">
                <span className="spec-name">Crop Damage Model (Model 2)</span>
                <span className="spec-info">7-Channel Dual-Temporal U-Net (Pre/Post B4, B8, NDVI, Delta-NDVI; BFCD-22 Benchmark)</span>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: XAI & INTERPRETABILITY */}
        {activeTab === 'interpretability' && (
          <div className="research-content-block">
            <div className="research-block-header">
              <h3>Explainable AI (XAI) &amp; Feature Attribution</h3>
              <p>
                Integrated Gradients and Occlusion Sensitivity analysis describing which input bands drove the network decision.
              </p>
            </div>

            <div className="xai-shares-grid">
              <div className="xai-bar-card">
                <div className="xai-share-header">
                  <span>VV/VH Ratio Polarization Band</span>
                  <strong className="cyan">53.4% Attribution Share</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill cyan" style={{ width: '53.4%' }} />
                </div>
                <p>
                  Elevated attribution share reflecting relative backscatter variation across smooth water surfaces vs rough vegetation.
                </p>
              </div>

              <div className="xai-bar-card">
                <div className="xai-share-header">
                  <span>VV Co-Polarization Backscatter</span>
                  <strong>29.8% Attribution Share</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: '29.8%', background: '#60a5fa' }} />
                </div>
                <p>
                  Primary co-polarized microwave signal reflecting calm specular water bodies.
                </p>
              </div>

              <div className="xai-bar-card">
                <div className="xai-share-header">
                  <span>VH Cross-Polarization Backscatter</span>
                  <strong>16.8% Attribution Share</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: '16.8%', background: '#94a3b8' }} />
                </div>
                <p>
                  Sensitive to volume scattering in emergent crops and urban building corners.
                </p>
              </div>
            </div>

            <div className="scientific-disclaimer-box">
              <strong>Epistemological Distinction: </strong>
              Feature attribution describes internal model gradient mechanics, NOT physical causation in nature. In modality ablation tests, VV alone achieves 0.5279 IoU on test chips; the flood signal resides primarily in absolute microwave backscatter.
            </div>
          </div>
        )}

        {/* TAB 4: DATASET & PROTOCOLS */}
        {activeTab === 'datasets' && (
          <div className="research-content-block">
            <div className="research-block-header">
              <h3>Dataset Partitioning &amp; Leave-One-Region-Out Protocol</h3>
              <p>
                Strict spatial partitioning across 446 hand-labeled chips to avoid spatial autocorrelation.
              </p>
            </div>

            <div className="dataset-splits-table">
              <div className="split-item">
                <div className="split-count">333</div>
                <div className="split-title">Training Chips</div>
                <div className="split-desc">Distributed globally across 11 flood events</div>
              </div>
              <div className="split-item">
                <div className="split-count">30</div>
                <div className="split-title">Validation Chips</div>
                <div className="split-desc">Mekong Basin (used strictly for epoch checkpoint selection)</div>
              </div>
              <div className="split-item highlight">
                <div className="split-count cyan">68</div>
                <div className="split-title">Held-Out India Test</div>
                <div className="split-desc">Zero training overlap; true unseen regional test set</div>
              </div>
              <div className="split-item">
                <div className="split-count">15</div>
                <div className="split-title">Reserved Chips</div>
                <div className="split-desc">Bolivia flood tiles held for future validation</div>
              </div>
            </div>

            <div className="manifest-links-box">
              <strong>Public Dataset Manifests Available in Repository:</strong>
              <ul>
                <li>• <code>data/manifests/sen1floods11_manifest.json</code> — CC BY 4.0</li>
                <li>• <code>data/manifests/bihar_fmis_manifest.json</code> — Open Government Data</li>
                <li>• <code>data/manifests/bhuvan_bihar_atlas_manifest.json</code> — NRSC / ISRO 1998–2019</li>
                <li>• <code>data/manifests/bfcd22_manifest.json</code> — Muzaffarpur Crop Damage Split</li>
                <li>• <code>data/manifests/floodnet_manifest.json</code> — Drone Benchmark (Supplementary)</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
