import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SAT-AI — Multi-Hazard Risk Research Platform',
  description:
    'Research prototype for multi-hazard risk assessment from satellite remote sensing, with provenance-enforced conversational analysis. Not an official warning system.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
