import type { BodyStateMessage, TelemetryState } from "../protocol/types";
import type {
  FaceCentroid,
  LocalVisionDebugFrame,
  LocalVisionDetection,
} from "../sensors/gaze";

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
  attachCameraPreview(video: HTMLVideoElement): void;
  setCameraUnavailable(): void;
  updateLocalVision(frame: LocalVisionDebugFrame): void;
  setObservationCaptureStatus(status: ObservationCaptureStatus): void;
  dispose(): void;
}

export const OBSERVATION_CAPTURE_STATUSES = Object.freeze([
  "waiting",
  "capturing",
  "sent",
  "error",
] as const);

export type ObservationCaptureStatus =
  (typeof OBSERVATION_CAPTURE_STATUSES)[number];

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
  "width:min(26rem,calc(100vw - 2rem))",
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

const CAPTURE_LABELS: Readonly<Record<ObservationCaptureStatus, string>> =
  Object.freeze({
    waiting: "OBSERVE JPEG · NOT REQUESTED",
    capturing: "OBSERVE JPEG · CAPTURING",
    sent: "OBSERVE JPEG · SENT TO CORE",
    error: "OBSERVE JPEG · FAILED",
  });

function isConnectionStatus(value: string): value is OverlayConnectionStatus {
  return OVERLAY_CONNECTION_STATUSES.some((status) => status === value);
}

function isCaptureStatus(value: string): value is ObservationCaptureStatus {
  return OBSERVATION_CAPTURE_STATUSES.some((status) => status === value);
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

function confidencePercent(value: number): string {
  return `${Math.round(Math.min(Math.max(value, 0), 1) * 100)}%`;
}

function positionPercent(value: number): string {
  return `${(Math.min(Math.max(value, 0), 1) * 100).toFixed(2)}%`;
}

function detectionBox(
  documentRef: Document,
  detection: LocalVisionDetection,
): HTMLElement {
  const box = element(
    documentRef,
    "div",
    `luxo-sensor-view__region luxo-sensor-view__region--${detection.kind}`,
  );
  const label = detection.kind === "face"
    ? `FACE ${confidencePercent(detection.confidence)}`
    : `HAND ${detection.index + 1} ${confidencePercent(detection.confidence)}`;
  box.dataset.kind = detection.kind;
  box.dataset.index = String(detection.index);
  box.dataset.confidence = detection.confidence.toFixed(3);
  box.style.left = positionPercent(detection.bounds.x);
  box.style.top = positionPercent(detection.bounds.y);
  box.style.width = positionPercent(detection.bounds.width);
  box.style.height = positionPercent(detection.bounds.height);
  box.append(element(documentRef, "span", "luxo-sensor-view__region-label", label));
  return box;
}

function targetPoint(
  documentRef: Document,
  centroid: FaceCentroid,
  kind: "face" | "hands",
): HTMLElement {
  const point = element(
    documentRef,
    "div",
    `luxo-sensor-view__target luxo-sensor-view__target--${kind}`,
  );
  point.dataset.target = kind;
  point.style.left = positionPercent(centroid.x);
  point.style.top = positionPercent(centroid.y);
  point.setAttribute("aria-label", `${kind} local attention point`);
  return point;
}

export class TelemetryOverlay implements TelemetryOverlayHandle {
  private readonly panel: HTMLElement;
  private readonly connectionValue: HTMLElement;
  private readonly disconnectedBadge: HTMLElement;
  private readonly fields: Readonly<Record<string, FieldElements>>;
  private readonly sensorView: HTMLElement;
  private readonly sensorViewport: HTMLElement;
  private readonly sensorAnnotations: HTMLElement;
  private readonly sensorFacts: HTMLElement;
  private readonly observationStatus: HTMLElement;
  private previewVideo: HTMLVideoElement | null = null;
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
    const sensorView = element(documentRef, "section", "luxo-sensor-view");
    const sensorHeader = element(documentRef, "header", "luxo-sensor-view__header");
    const sensorTitle = element(documentRef, "span", "luxo-sensor-view__title", "LOCAL SENSOR VIEW");
    const sensorScope = element(documentRef, "span", "luxo-sensor-view__scope", "FACE + HANDS ONLY");
    const sensorViewport = element(documentRef, "div", "luxo-sensor-view__viewport");
    const sensorAnnotations = element(documentRef, "div", "luxo-sensor-view__annotations");
    const sensorFacts = element(documentRef, "p", "luxo-sensor-view__facts", "LOCAL DETECTION · WAITING");
    const observationStatus = element(
      documentRef,
      "p",
      "luxo-sensor-view__capture",
      CAPTURE_LABELS.waiting,
    );

    panel.style.cssText = PANEL_STYLE;
    header.style.cssText = HEADER_STYLE;
    disconnectedBadge.style.cssText = BADGE_STYLE;
    grid.style.cssText = GRID_STYLE;
    panel.dataset.component = "luxo-telemetry";
    panel.setAttribute("aria-label", "Luxo live telemetry");
    panel.setAttribute("aria-live", "polite");
    disconnectedBadge.setAttribute("role", "alert");
    disconnectedBadge.dataset.role = "disconnected-badge";
    sensorView.hidden = true;
    sensorView.dataset.component = "luxo-sensor-view";
    sensorView.dataset.camera = "unavailable";
    sensorView.dataset.capture = "waiting";
    sensorView.setAttribute("aria-label", "Local camera analysis preview");
    sensorView.setAttribute("aria-live", "off");
    sensorViewport.setAttribute("aria-label", "Mirrored camera frame with local detections");

    header.append(title, connectionValue);
    sensorHeader.append(sensorTitle, sensorScope);
    sensorViewport.append(sensorAnnotations);
    sensorView.append(sensorHeader, sensorViewport, sensorFacts, observationStatus);
    panel.append(header, disconnectedBadge, grid, sensorView);
    root.append(panel);

    this.panel = panel;
    this.connectionValue = connectionValue;
    this.disconnectedBadge = disconnectedBadge;
    this.sensorView = sensorView;
    this.sensorViewport = sensorViewport;
    this.sensorAnnotations = sensorAnnotations;
    this.sensorFacts = sensorFacts;
    this.observationStatus = observationStatus;
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

  attachCameraPreview(video: HTMLVideoElement): void {
    this.assertMounted();
    if (this.previewVideo && this.previewVideo !== video) {
      this.previewVideo.hidden = true;
      this.previewVideo.remove();
    }
    this.previewVideo = video;
    video.className = "luxo-sensor-view__video";
    video.hidden = false;
    video.muted = true;
    video.playsInline = true;
    video.setAttribute("aria-label", "Live mirrored local camera preview");
    const width = video.videoWidth;
    const height = video.videoHeight;
    this.sensorViewport.style.aspectRatio =
      Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0
        ? `${width} / ${height}`
        : "";
    this.sensorViewport.prepend(video);
    this.sensorView.hidden = false;
    this.sensorView.dataset.camera = "live";
    this.sensorFacts.textContent = "LOCAL DETECTION · FACE — · HANDS —";
  }

  setCameraUnavailable(): void {
    if (this.disposed) return;
    if (this.previewVideo) {
      this.previewVideo.hidden = true;
      this.previewVideo.remove();
      this.previewVideo = null;
    }
    this.sensorAnnotations.replaceChildren();
    this.sensorFacts.textContent = "LOCAL DETECTION · CAMERA UNAVAILABLE";
    this.sensorView.dataset.camera = "unavailable";
    this.sensorView.hidden = true;
  }

  updateLocalVision(frame: LocalVisionDebugFrame): void {
    this.assertMounted();
    if (this.sensorView.hidden) return;
    const detections = [
      ...(frame.face ? [frame.face] : []),
      ...frame.hands,
    ];
    const annotations = detections.map((detection) =>
      detectionBox(this.sensorView.ownerDocument, detection));
    if (frame.face) {
      annotations.push(targetPoint(this.sensorView.ownerDocument, frame.face.centroid, "face"));
    }
    if (frame.handCentroid) {
      annotations.push(targetPoint(this.sensorView.ownerDocument, frame.handCentroid, "hands"));
    }
    this.sensorAnnotations.replaceChildren(...annotations);
    const face = frame.face ? confidencePercent(frame.face.confidence) : "—";
    this.sensorFacts.textContent = `LOCAL DETECTION · FACE ${face} · HANDS ${frame.hands.length}`;
    this.sensorView.dataset.face = frame.face ? "present" : "absent";
    this.sensorView.dataset.hands = String(frame.hands.length);
  }

  setObservationCaptureStatus(status: ObservationCaptureStatus): void {
    this.assertMounted();
    if (!isCaptureStatus(status)) {
      throw new RangeError(`Unknown observation capture status: ${status}`);
    }
    this.sensorView.dataset.capture = status;
    this.observationStatus.textContent = CAPTURE_LABELS[status];
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.facts = null;
    if (this.previewVideo) {
      this.previewVideo.hidden = true;
      this.previewVideo.remove();
      this.previewVideo = null;
    }
    this.sensorAnnotations.replaceChildren();
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
