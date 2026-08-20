# Luxo

Luxo is a five-degree-of-freedom animated desk lamp. It uses the laptop camera and microphone, then responds with motion, light, speech, sound effects, and music.

## Run locally

Run these commands from the repository root.

### 1. Check the requirements

- Ubuntu 24.04 or macOS
- CPython 3.12
- Node.js `^20.19.0 || >=22.12.0`
- A paid OpenRouter API key

On Ubuntu, the setup script installs the required system packages. On macOS, it checks for the required tools and tells you what is missing. A typical macOS setup uses:

```sh
xcode-select --install
brew install cmake python@3.12 node espeak-ng
```

If the Xcode Command Line Tools are already installed, skip the first command.

### 2. Install the project

```sh
./setup.sh
```

The script creates `.venv`, installs the pinned Python and Node dependencies, builds whisper.cpp, downloads the required models and renderer assets, and builds the renderer. It is safe to run again if setup is interrupted.

### 3. Configure OpenRouter

Create a file named `.env` in the repository root:

```dotenv
OPENROUTER_API_KEY=replace_with_your_paid_key
OPENROUTER_MODEL=google/gemini-2.5-flash-lite:nitro
```

Then restrict access to the file:

```sh
chmod 600 .env
```

A paid OpenRouter key is required for Luxo to work as intended. The free OpenRouter models do not support the image input used by the lamp. `google/gemini-2.5-flash-lite:nitro` is the model used during development and testing, and is the recommended model for running the project.

### 4. Run Luxo

```sh
./run.sh
```

`run.sh` loads `.env`, runs the local checks, and starts both the Python core and browser renderer. Once both are ready, open:

- Main interface: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- Self-test page: [http://127.0.0.1:5173/selftest](http://127.0.0.1:5173/selftest)

Allow camera and microphone access when the browser asks. Press `Ctrl-C` in the terminal to stop both processes.

To verify the OpenRouter key before starting Luxo, run:

```sh
.venv/bin/python doctor.py --check-openrouter
```

This optional check makes a small authenticated request to OpenRouter.

For the production-style renderer:

```sh
./run.sh --prod
```

This serves the built renderer at [http://127.0.0.1:4173](http://127.0.0.1:4173).

## Submission documents

The required submission documents are in the repository root:

| Document | What it contains |
| --- | --- |
| [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md) | Architecture and data flow, protocol, model-to-action design, simulation, deployment, measurements, trade-offs, and known limitations |
| [SUBMISSION.md](SUBMISSION.md) | Submission requirements, expected deliverables, and evaluation criteria |
| [CHALLENGE.md](CHALLENGE.md) | The original challenge brief, constraints, and demonstration requirements |
