/** @type {import('next').NextConfig} */
const backendUrl =
  process.env.NEXT_PUBLIC_API_URL ||
  'https://sat-ai-api-chiragpednekar3-8808s-projects.vercel.app';

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/chat',
        destination: `${backendUrl}/api/v1/chat`,
      },
      {
        source: '/api/v1/:path*',
        destination: `${backendUrl}/api/v1/:path*`,
      },
      {
        source: '/health',
        destination: `${backendUrl}/health`,
      },
    ];
  },
};

export default nextConfig;
