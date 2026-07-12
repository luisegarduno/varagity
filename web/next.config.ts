import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server bundle for the compose image (web/Dockerfile
  // copies .next/standalone + .next/static + public).
  output: "standalone",
};

export default nextConfig;
