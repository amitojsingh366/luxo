import type { BodyStateMessage, TelemetryState } from "../protocol/types";

export const OVERLAY_CONNECTION_STATUSES = Object.freeze([
  "connecting",
  "connected",
  "disconnected",
] as const);

export type OverlayConnectionStatus =
  (typeof OVERLAY_CONNECTION_STATUSES)[number];

export interface TelemetryOverlayViewModel {
  readonly connection: OverlayConnectionStatus;
  readonly connectionLabel: string;
  readonly disconnected: boolean;
  readonly state: string;
  readonly gazePresence: string;
  readonly gazePresentValue: string;
  readonly gazeYaw: string;
  readonly gazePitch: string;
  readonly planDepth: string;
  readonly memoryCount: string;
  readonly latency: string;
  readonly velocityClamps: string;
  readonly limitClamps: string;
}

export interface TelemetryOverlayHandle {
  setConnectionStatus(status: OverlayConnectionStatus): void;
  updateBodyState(state: Pick<BodyStateMessage, "telemetry">): void;
  updateFacts(facts: TelemetryState): void;
  dispose(): void;
}

const EMPTY_VALUE = "—";
const CONNECTION_LABELS: Readonly<Record<OverlayConnectionStatus, string>> =
  Object.freeze({
    connecting: "CORE CONNECTING",
    connected: "CORE CONNECTED",
    disconnected: "CORE DISCONNECTED",
  });

const PANEL_STYLE = [
  "position:absolute",
  "z-index:20",
  "top:clamp(1rem,3vw,2rem)",
  "right:clamp(1rem,3vw,2rem)",
  "width:min(19rem,calc(100% - 2rem))",
  "padding:0.8rem",
  "border:1px solid rgba(255,255,255,0.12)",
  "border-radius:0.8rem",
  "color:#f4efe7",
  "background:rgba(8,11,17,0.78)",
  "box-shadow:0 1rem 3rem rgba(0,0,0,0.32)",
  "font:600 0.68rem/1.35 ui-monospace,SFMono-Regular,Menlo,monospace",
  "letter-spacing:0.04em",
  "pointer-events:none",
  "backdrop-filter:blur(14px)",
].join(";");

const HEADER_STYLE = [
  "display:flex",
  "justify-content:space-between",
  "gap:0.75rem",
  "margin-bottom:0.65rem",
  "color:rgba(244,239,231,0.7)",
  "font-size:0.62rem",
  "letter-spacing:0.12em",
].join(";");

const BADGE_STYLE = [
  "margin-bottom:0.65rem",
  "padding:0.55rem 0.65rem",
  "border:1px solid #ff8b7d",
  "border-radius:0.45rem",
  "color:#fff7f5",
  "background:#b42318",
  "box-shadow:0 0 1.5rem rgba(255,91,73,0.45)",
  "font-weight:800",
  "letter-spacing:0.1em",
  "text-align:center",
].join(";");

const GRID_STYLE = [
  "display:grid",
  "grid-template-columns:1fr auto",
  "gap:0.35rem 0.9rem",
  "margin:0",
].join(";");

const LABEL_STYLE = "margin:0;color:rgba(244,239,231,0.54);font-weight:500";
const VALUE_STYLE = "margin:0;color:#fffaf2;text-align:right;font-variant-numeric:tabular-nums";

function isConnectionStatus(value: string): value is OverlayConnectionStatus {
  return OVERLAY_CONNECTION_STATUSES.some((status) => status === value);
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function formatOverlayCount(value: unknown): string {
  const number = finiteNumber(value);
  if (number === null || number < 0) return EMPTY_VALUE;
  return Math.trunc(number).toLocaleString("en-US", { useGrouping: false });
}

export function formatOverlayLatency(value: unknown): string {
  const number = finiteNumber(value);
  if (number === null || number < 0) return EMPTY_VALUE;
  return `${number.toFixed(0)} ms`;
}

export function formatOverlayDegrees(value: unknown): string {
  const number = finiteNumber(value);
  if (number === null) return EMPTY_VALUE;
  const sign = number > 0 ? "+" : number < 0 ? "−" : "";
  return `${sign}${Math.abs(number).toFixed(1)}°`;
}

function formatState(value: unknown): string {
  return typeof value === "string" && value.length > 0 ? value : "UNKNOWN";
}

function formatGazePresence(value: unknown): {
  label: string;
  value: string;
} {
  if (value === true) return { label: "PRESENT", value: "true" };
  if (value === false) return { label: "ABSENT", value: "false" };
  return { label: "UNKNOWN", value: "unknown" };
}

export function buildTelemetryOverlayViewModel(
  connection: OverlayConnectionStatus,
  facts: TelemetryState | null,
): TelemetryOverlayViewModel {
  if (!isConnectionStatus(connection)) {
    throw new RangeError(`Unknown overlay connection status: ${connection}`);
  }
  const gaze = formatGazePresence(facts?.gaze.present);
  return Object.freeze({
    connection,
    connectionLabel: CONNECTION_LABELS[connection],
    disconnected: connection === "disconnected",
    state: formatState(facts?.state),
    gazePresence: gaze.label,
    gazePresentValue: gaze.value,
    gazeYaw: formatOverlayDegrees(facts?.gaze.yaw_deg),
    gazePitch: formatOverlayDegrees(facts?.gaze.pitch_deg),
    planDepth: formatOverlayCount(facts?.plan_depth),
    memoryCount: formatOverlayCount(facts?.memory_count),
    latency: formatOverlayLatency(facts?.last_latency_ms),
    velocityClamps: formatOverlayCount(facts?.clamps.vel),
    limitClamps: formatOverlayCount(facts?.clamps.limit),
  });
}

interface FieldElements {
  readonly value: HTMLElement;
}

function element(
  documentRef: Document,
  tag: keyof HTMLElementTagNameMap,
  className: string,
  text = "",
): HTMLElement {
  const node = documentRef.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
}

function addField(
  documentRef: Document,
  grid: HTMLElement,
  key: string,
  label: string,
): FieldElements {
  const term = element(documentRef, "dt", "luxo-telemetry__label", label);
  const value = element(documentRef, "dd", "luxo-telemetry__value", EMPTY_VALUE);
  term.style.cssText = LABEL_STYLE;
  value.style.cssText = VALUE_STYLE;
  value.dataset.field = key;
  value.dataset.value = "unknown";
  grid.append(term, value);
  return { value };
}

export class TelemetryOverlay implements TelemetryOverlayHandle {
  private readonly panel: HTMLElement;
  private readonly connectionValue: HTMLElement;
  private readonly disconnectedBadge: HTMLElement;
  private readonly fields: Readonly<Record<string, FieldElements>>;
  private connection: OverlayConnectionStatus = "connecting";
  private facts: TelemetryState | null = null;
  private disposed = false;

  constructor(root: HTMLElement, documentRef: Document = root.ownerDocument) {
    const panel = element(documentRef, "section", "luxo-telemetry");
    const header = element(documentRef, "header", "luxo-telemetry__header");
    const title = element(documentRef, "span", "luxo-telemetry__title", "LUXO TELEMETRY");
    const connectionValue = element(documentRef, "span", "luxo-telemetry__connection");
    const disconnectedBadge = element(
      documentRef,
      "div",
      "luxo-telemetry__disconnected",
      "CORE DISCONNECTED",
    );
    const grid = element(documentRef, "dl", "luxo-telemetry__grid");

    panel.style.cssText = PANEL_STYLE;
    header.style.cssText = HEADER_STYLE;
    disconnectedBadge.style.cssText = BADGE_STYLE;
    grid.style.cssText = GRID_STYLE;
    panel.dataset.component = "luxo-telemetry";
    panel.setAttribute("aria-label", "Luxo live telemetry");
    panel.setAttribute("aria-live", "polite");
    disconnectedBadge.setAttribute("role", "alert");
    disconnectedBadge.dataset.role = "disconnected-badge";

    header.append(title, connectionValue);
    panel.append(header, disconnectedBadge, grid);
    root.append(panel);

    this.panel = panel;
    this.connectionValue = connectionValue;
    this.disconnectedBadge = disconnectedBadge;
    this.fields = Object.freeze({
      state: addField(documentRef, grid, "state", "FSM STATE"),
      gazePresence: addField(documentRef, grid, "gaze-present", "GAZE"),
      gazeYaw: addField(documentRef, grid, "gaze-yaw-deg", "GAZE YAW"),
      gazePitch: addField(documentRef, grid, "gaze-pitch-deg", "GAZE PITCH"),
      planDepth: addField(documentRef, grid, "plan-depth", "PLAN DEPTH"),
      memoryCount: addField(documentRef, grid, "memory-count", "MEMORY OBJECTS"),
      latency: addField(documentRef, grid, "last-latency-ms", "MODEL LATENCY"),
      velocityClamps: addField(documentRef, grid, "velocity-clamps", "VELOCITY CLAMPS"),
      limitClamps: addField(documentRef, grid, "limit-clamps", "LIMIT CLAMPS"),
    });
    this.render();
  }

  setConnectionStatus(status: OverlayConnectionStatus): void {
    this.assertMounted();
    if (!isConnectionStatus(status)) {
      throw new RangeError(`Unknown overlay connection status: ${status}`);
    }
    this.connection = status;
    this.render();
  }

  updateBodyState(state: Pick<BodyStateMessage, "telemetry">): void {
    this.updateFacts(state.telemetry);
  }

  updateFacts(facts: TelemetryState): void {
    this.assertMounted();
    this.facts = facts;
    this.render();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.facts = null;
    this.panel.remove();
  }

  private render(): void {
    const view = buildTelemetryOverlayViewModel(this.connection, this.facts);
    this.panel.dataset.connection = view.connection;
    this.panel.dataset.state = view.state;
    this.panel.dataset.gazePresent = view.gazePresentValue;
    this.panel.dataset.memoryCount = view.memoryCount;
    this.connectionValue.textContent = view.connectionLabel;
    this.connectionValue.dataset.field = "connection";
    this.connectionValue.dataset.value = view.connection;
    this.disconnectedBadge.hidden = !view.disconnected;
    this.disconnectedBadge.style.display = view.disconnected ? "block" : "none";
    this.setField("state", view.state);
    this.setField("gazePresence", view.gazePresence, view.gazePresentValue);
    this.setField("gazeYaw", view.gazeYaw);
    this.setField("gazePitch", view.gazePitch);
    this.setField("planDepth", view.planDepth);
    this.setField("memoryCount", view.memoryCount);
    this.setField("latency", view.latency);
    this.setField("velocityClamps", view.velocityClamps);
    this.setField("limitClamps", view.limitClamps);
  }

  private setField(key: string, text: string, value = text): void {
    const field = this.fields[key];
    if (field === undefined) throw new Error(`Missing telemetry field: ${key}`);
    field.value.textContent = text;
    field.value.dataset.value = value;
  }

  private assertMounted(): void {
    if (this.disposed) throw new Error("Telemetry overlay has been disposed");
  }
}

export function mountTelemetryOverlay(root: HTMLElement): TelemetryOverlayHandle {
  return new TelemetryOverlay(root);
}
