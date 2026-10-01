import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SAT-AI — AI-Powered Flood Detection from Satellite Imagery',
  description:
    'SAT-AI uses Sentinel-1 SAR imagery and deep learning to identify flood-water extent, evaluate model performance on unseen geographic regions, and provide grounded explanations through an AI assistant.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" sizes="any" />
      </head>
      <body>{children}</body>
    </html>
  );
}
