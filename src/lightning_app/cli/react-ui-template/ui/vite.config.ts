import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // NOTE: Component UI's are served under `/{componentName}/` subpath, so the app needs to be configured for relative base path.
  base: "./",
});
