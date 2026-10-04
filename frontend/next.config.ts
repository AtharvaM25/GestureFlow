import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // self-contained server bundle for the Docker image (docker/frontend.Dockerfile)
  output: "standalone",
};

export default nextConfig;
