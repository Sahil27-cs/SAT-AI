'use client';

import React from 'react';

/**
 * FormattedAnswer: Converts scientific markdown output from SAT-AI into
 * clean, styled semantic React elements without needing heavy external dependencies.
 * Handles headings (###, ####), bullet points, bold emphasis, inline code,
 * and caveat callouts with high aesthetic polish.
 */
export function FormattedAnswer({ text }: { text: string }) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: string[] = [];
  let listKey = 0;

  const flushList = () => {
    if (currentList.length > 0) {
      elements.push(
        <ul key={`list-${listKey++}`} className="formatted-list">
          {currentList.map((item, idx) => (
            <li key={idx}>{renderInlineMarkdown(item)}</li>
          ))}
        </ul>
      );
      currentList = [];
    }
  };

  lines.forEach((line, i) => {
    const trimmed = line.trim();

    if (!trimmed) {
      flushList();
      return;
    }

    // Level 3 Heading: ### Heading
    if (trimmed.startsWith('### ')) {
      flushList();
      const content = trimmed.slice(4);
      elements.push(
        <h3 key={`h3-${i}`} className="formatted-h3">
          <span className="h3-icon">✦</span>
          {content}
        </h3>
      );
      return;
    }

    // Level 4 Heading: #### Heading
    if (trimmed.startsWith('#### ')) {
      flushList();
      const content = trimmed.slice(5);
      elements.push(
        <h4 key={`h4-${i}`} className="formatted-h4">
          <span className="h4-icon">↳</span>
          {renderInlineMarkdown(content)}
        </h4>
      );
      return;
    }

    // Bullet points: - item or * item
    if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
      currentList.push(trimmed.slice(2));
      return;
    }

    // Indented sub-bullets
    if (line.startsWith('  - ') || line.startsWith('    - ')) {
      currentList.push(line.replace(/^\s+-\s+/, ''));
      return;
    }

    // Refusal / Caveat alert block
    if (
      trimmed.includes('SAT-AI cannot advise') ||
      trimmed.includes('SAT-AI performs no earthquake prediction') ||
      trimmed.includes('SAT-AI does not forecast cyclone') ||
      trimmed.includes('SAT-AI does not predict') ||
      trimmed.includes('Language layer unavailable') ||
      trimmed.includes('REJECTED (Blocked)')
    ) {
      flushList();
      elements.push(
        <div key={`alert-${i}`} className="formatted-alert">
          <div className="alert-badge">GOVERNANCE &amp; SAFETY VERDICT</div>
          <div className="alert-content">{renderInlineMarkdown(trimmed)}</div>
        </div>
      );
      return;
    }

    flushList();
    elements.push(
      <p key={`p-${i}`} className="formatted-p">
        {renderInlineMarkdown(trimmed)}
      </p>
    );
  });

  flushList();

  return <div className="formatted-answer">{elements}</div>;
}

/**
 * Replaces **bold** and `code` with proper React spans and code blocks.
 */
function renderInlineMarkdown(str: string): React.ReactNode {
  // Regex splitting by bold (**...**) or code (`...`)
  const parts = str.split(/(\*\*[^*]+\*\*|`[^`]+`|\$[^$]+\$)/g);

  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={index} className="formatted-bold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={index} className="formatted-code">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith('$') && part.endsWith('$')) {
      return (
        <span key={index} className="formatted-math">
          {part.slice(1, -1)}
        </span>
      );
    }
    return part;
  });
}
