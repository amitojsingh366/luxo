import { mountRenderer } from './app';
import { ProtocolClient } from './protocol/client';

async function main(): Promise<void> {
  const root = document.querySelector<HTMLElement>('#app');
  if (!root) throw new Error('Missing #app mount element');

  const renderer = await mountRenderer(root);
  const protocol = new ProtocolClient({
    url: 'ws://127.0.0.1:8765',
    hello: {
      type: 'hello',
      fps: 60,
      camera: { w: 640, h: 480, hfov_deg: 60 },
    },
    onBodyState: (state) => renderer.applyBodyState(state),
    onError: (error) => console.error(error),
  });

  protocol.connect();
  window.addEventListener(
    'beforeunload',
    () => {
      protocol.disconnect();
      renderer.destroy();
    },
    { once: true },
  );
}

void main();
