'use client';

import Link from 'next/link';
import { ResearchSection } from '@/components/ResearchSection';

export default function ResearchPage() {
  return (
    <>
      <header className="site-header">
        <div className="site-container nav-container">
          <Link href="/" className="brand-link">
            <div className="brand-icon">
              <span className="brand-icon-pulse" />
            </div>
            <div className="brand-text">
              <span className="brand-title">SAT-AI</span>
              <span className="brand-sub">Research &amp; Scientific Methodology</span>
            </div>
          </Link>

          <nav>
            <ul className="nav-links">
              <li>
                <Link href="/" className="nav-link">
                  ← Back to Place Explorer
                </Link>
              </li>
              <li>
                <Link href="/#disasters" className="nav-link">
                  Disasters
                </Link>
              </li>
              <li>
                <Link href="/#assistant" className="nav-link">
                  Ask SAT-AI
                </Link>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main className="site-container" style={{ paddingTop: '40px' }}>
        <ResearchSection />
      </main>

      <footer className="site-footer">
        <div className="site-container footer-container">
          <div className="footer-left">
            <div className="footer-brand">SAT-AI Research Portal</div>
            <div className="footer-tag">Empirical satellite remote sensing and machine learning methodology.</div>
            <div className="footer-disclaimer">
              Student research prototype — not an official warning system.
            </div>
          </div>
          <ul className="footer-links">
            <li>
              <Link href="/" className="footer-link">
                Home
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
