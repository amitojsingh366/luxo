import { mountLuxoBrowserRuntime } from "./runtime";

async function main(): Promise<void> {
  const root = document.querySelector<HTMLElement>("#app");
  if (!root) throw new Error("Missing #app mount element");
  const runtime = await mountLuxoBrowserRuntime(root);
  window.addEventListener(
    "beforeunload",
    () => { void runtime.destroySafely(); },
    { once: true },
  );
}

void main().catch((error: unknown) => console.error(error));
