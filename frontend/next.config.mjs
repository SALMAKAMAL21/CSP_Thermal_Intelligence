import constants from "next/constants.js";

/** @param {string} phase @returns {import('next').NextConfig} */
const nextConfig = (phase) => ({
  reactStrictMode: true,
  // Les serveurs dev et les compilations de production ne doivent pas écraser leurs fichiers.
  distDir: phase === constants.PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next"
});

export default nextConfig;
