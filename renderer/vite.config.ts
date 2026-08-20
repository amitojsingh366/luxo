import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const repositoryRoot = fileURLToPath(new URL("..", import.meta.url));
const mainPage = fileURLToPath(new URL("index.html", import.meta.url));
const selftestPage = fileURLToPath(new URL("selftest.html", import.meta.url));

// Vite treats whatever these hooks return as a post hook and calls it with no arguments once its
// own middlewares are installed. Connect's `middlewares.use()` returns the app for chaining, so a
// concise arrow body here hands Vite the connect app and it dies calling `app()` at startup. Both
// hooks must use a block body and return nothing.
function selftestRoute(): Plugin {
  const rewrite = () => (request: { url?: string }, _response: unknown, next: () => void) => {
    if (request.url?.split("?", 1)[0] === "/selftest") request.url = "/selftest.html";
    next();
  };
  return {
    name: "luxo-selftest-route",
    configureServer: (server) => {
      server.middlewares.use(rewrite());
    },
    configurePreviewServer: (server) => {
      server.middlewares.use(rewrite());
    },
  };
}

export default defineConfig({
  assetsInclude: ["**/*.stl", "**/*.urdf"],
  plugins: [selftestRoute()],
  build: {
    rollupOptions: {
      input: { main: mainPage, selftest: selftestPage },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    fs: {
      allow: [repositoryRoot],
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
});
