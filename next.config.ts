import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Both routes prerender as static content and all the CNN work runs in the
  // browser, so there is nothing to server-render. Emits out/ for Firebase.
  output: "export",
};

export default nextConfig;
