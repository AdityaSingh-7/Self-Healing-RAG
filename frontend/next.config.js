/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow API calls to our FastAPI backend
  async rewrites() {
    return [
      {
        source: "/api/backend/:path*",
        destination: `${process.env.BACKEND_URL || "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
