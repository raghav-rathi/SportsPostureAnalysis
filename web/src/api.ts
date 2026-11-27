import type {
  CompareStory,
  CoachingStory,
  DeliveryCard,
  JobStatus,
  OverviewResponse,
  ProgressResponse,
  ScanCandidate,
  BowlerSummary,
  CoachNote,
  DeliveryDetailResponse,
  HealthResponse,
  SessionSummary,
  TelestratorShape,
  ArchetypesResponse,
  ExportFormat,
  SpellAnalyticsResponse,
  WorkloadLog,
  WorkloadLogRequest,
  WorkloadResponse,
} from "./types";

const BASE = "";

async function jsonReq<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    let detail = "";
    try {
      const parsed = await res.json();
      detail = parsed.detail || JSON.stringify(parsed);
    } catch {
      detail = await res.text();
    }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => jsonReq<HealthResponse>("/api/health"),
  listBowlers: () => jsonReq<{ bowlers: BowlerSummary[] }>("/api/bowlers"),
  listSessions: () => jsonReq<{ sessions: SessionSummary[] }>("/api/sessions"),
  getSession: (id: string) => jsonReq<{ session: SessionSummary }>(`/api/sessions/${id}`),
  createSession: (body: {
    bowler_id?: string;
    bowler_name?: string;
    bowling_arm?: string;
    age_band?: string;
    title: string;
    notes?: string;
  }) =>
    jsonReq<{ session: SessionSummary }>("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getDelivery: (id: string) => jsonReq<DeliveryDetailResponse>(`/api/deliveries/${id}`),
  deleteDelivery: (id: string) =>
    jsonReq<{ deleted: boolean; delivery_id: string }>(`/api/deliveries/${id}`, { method: "DELETE" }),
  retrackBowler: (id: string, anchor_xy_norm: [number, number]) =>
    jsonReq<any>(`/api/deliveries/${id}/retrack-bowler`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ anchor_xy_norm }),
    }),
  getAnnotations: (id: string) =>
    jsonReq<{ delivery_id: string; annotations: TelestratorShape[] }>(`/api/deliveries/${id}/annotations`),
  saveAnnotations: (id: string, annotations: TelestratorShape[]) =>
    jsonReq<{ delivery_id: string; annotations: TelestratorShape[] }>(`/api/deliveries/${id}/annotations`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotations }),
    }),
  listNotes: (id: string) => jsonReq<{ delivery_id: string; notes: CoachNote[] }>(`/api/deliveries/${id}/notes`),
  addNote: (id: string, body: { frame_idx?: number | null; timestamp_s?: number | null; tag: string; body: string }) =>
    jsonReq<{ note: CoachNote }>(`/api/deliveries/${id}/notes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteNote: (id: string, noteId: string) =>
    jsonReq<{ deleted: boolean; note_id: string }>(`/api/deliveries/${id}/notes/${noteId}`, { method: "DELETE" }),
  previewCalibration: (id: string, mcc_landmarks_px: Record<string, [number, number]>) =>
    jsonReq<{ calibration: any; mcc_wireframe_2d: Record<string, [number, number]>; reprojection_error_px: number | null }>(
      `/api/deliveries/${id}/preview-calibration`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mcc_landmarks_px }),
      }
    ),
  recomputeDelivery: (id: string, payload: Record<string, any>) =>
    jsonReq<{
      recompute_ms: number;
      needs_gpu_refit_for_K: boolean;
      fact_sheet: DeliveryDetailResponse["fact_sheet"];
      telemetry: DeliveryDetailResponse["telemetry"];
      report_markdown: string;
      isb_summary: DeliveryDetailResponse["delivery"]["isb_summary"];
      pan_summary: DeliveryDetailResponse["delivery"]["pan_summary"];
    }>(`/api/deliveries/${id}/recompute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  askCoach: (id: string, question: string, use_external_llm = false) =>
    jsonReq<{
      answer_markdown: string;
      cited_metric_keys: string[];
      verification_passed: boolean;
      blocked_violations: string[];
      generator_mode: string;
    }>(`/api/deliveries/${id}/coach/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, use_external_llm }),
    }),
  exportReel: (id: string) =>
    jsonReq<{ reel_url: string; size_bytes: number }>(`/api/deliveries/${id}/export-reel`, { method: "POST" }),
  compareSession: (sessionId: string) => jsonReq<any>(`/api/sessions/${sessionId}/compare`),
  fuseDual: (sessionId: string, delivery_id_a: string, delivery_id_b: string, label: string) =>
    jsonReq<any>(`/api/sessions/${sessionId}/fuse-dual`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ delivery_id_a, delivery_id_b, label }),
    }),
  getWorkload: (bowlerId: string) => jsonReq<any>(`/api/bowlers/${bowlerId}/workload`),
  addWorkload: (bowlerId: string, body: { log_date: string; deliveries_count: number; session_type: string; notes: string }) =>
    jsonReq<any>(`/api/bowlers/${bowlerId}/workload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  scanVideo: (sessionId: string, file: File) => {
    const fd = new FormData();
    fd.append("video", file);
    return jsonReq<{
      scan_id: string;
      upload_path: string;
      video_meta: any;
      candidates: Array<{
        candidate_id: number;
        start_frame: number;
        end_frame: number;
        peak_frame: number;
        start_s: number;
        end_s: number;
        motion_score: number;
        thumbnail_b64?: string | null;
      }>;
    }>(`/api/sessions/${sessionId}/scan-video`, { method: "POST", body: fd });
  },
  importFromScan: (
    sessionId: string,
    payload: {
      scan_id: string;
      clips: Array<{ start_frame: number; end_frame?: number | null; label: string }>;
      bowling_arm?: string | null;
      capture_fps_override?: number | null;
    }
  ) =>
    jsonReq<{ scan_id: string; deliveries: Array<{ delivery_id: string; label: string }> }>(
      `/api/sessions/${sessionId}/deliveries/from-scan`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  uploadDelivery: (sessionId: string, fd: FormData) =>
    jsonReq<{ delivery_id: string; recompute_ms: number; fact_sheet: DeliveryDetailResponse["fact_sheet"] }>(
      `/api/sessions/${sessionId}/deliveries/upload`,
      { method: "POST", body: fd }
    ),

  // ── Studio v4 consumer endpoints ───────────────────────────────────────────
  overview: () => jsonReq<OverviewResponse>("/api/overview"),
  frameUrl: (deliveryId: string, frameIdx: number, w = 480) =>
    `${BASE}/api/deliveries/${deliveryId}/frame/${Math.max(0, Math.round(frameIdx))}.jpg?w=${w}`,
  patchDelivery: (id: string, body: { label?: string }) =>
    jsonReq<{ delivery: DeliveryCard; story?: CoachingStory }>(`/api/deliveries/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  compare: (a: string, b: string) =>
    jsonReq<CompareStory>(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`),
  progress: (bowlerId: string) => jsonReq<ProgressResponse>(`/api/bowlers/${bowlerId}/progress`),
  createBowler: (body: { name: string; bowling_arm: "right" | "left"; age_band: string }) =>
    jsonReq<{ bowler: BowlerSummary }>("/api/bowlers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  bowlerScanVideo: (bowlerId: string, file: File) => {
    const fd = new FormData();
    fd.append("video", file);
    return jsonReq<{ scan_id: string; session_id: string; video_meta: any; candidates: ScanCandidate[] }>(
      `/api/bowlers/${bowlerId}/scan-video`,
      { method: "POST", body: fd }
    );
  },
  bowlerImportFromScan: (
    bowlerId: string,
    payload: {
      scan_id: string;
      clips: Array<{ start_frame: number; end_frame?: number | null; label: string }>;
      bowling_arm?: string | null;
      capture_fps_override?: number | null;
    }
  ) =>
    jsonReq<{ scan_id: string; session_id?: string; job_id?: string | null; deliveries: Array<{ delivery_id: string; label: string }> }>(
      `/api/bowlers/${bowlerId}/deliveries/from-scan`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
    ),
  bowlerUploadDelivery: (bowlerId: string, fd: FormData) =>
    jsonReq<{ delivery_id: string; session_id?: string; job_id?: string | null; story?: CoachingStory | null }>(
      `/api/bowlers/${bowlerId}/deliveries/upload`,
      { method: "POST", body: fd }
    ),
  job: (jobId: string) => jsonReq<JobStatus>(`/api/jobs/${jobId}`),

  // ── Studio v5 endpoints (docs/research_app_landscape_v5.md §10) ────────────
  /** View-adaptive multi-exposure JPEG (panoramic_composite on side-on, coronal_sequence_strip on rear/front). */
  stroboscopicUrl: (deliveryId: string, opts: { w?: number; skeleton?: boolean; trails?: boolean } = {}) =>
    `${BASE}/api/deliveries/${deliveryId}/stroboscopic.jpg?w=${opts.w ?? 1280}&skeleton=${opts.skeleton ? "true" : "false"}&trails=${
      opts.trails === false ? "false" : "true"
    }`,
  videoUrl: (deliveryId: string) => `${BASE}/api/deliveries/${deliveryId}/video`,
  exportUrl: (deliveryId: string, format: ExportFormat = "csv") =>
    `${BASE}/api/deliveries/${deliveryId}/export?format=${format}`,
  archetypes: (arm: "right" | "left" = "right") => jsonReq<ArchetypesResponse>(`/api/archetypes?arm=${arm}`),
  workload: (bowlerId: string) => jsonReq<WorkloadResponse>(`/api/bowlers/${bowlerId}/workload`),
  logWorkload: (bowlerId: string, body: WorkloadLogRequest) =>
    jsonReq<{ log: WorkloadLog } & Partial<WorkloadResponse>>(`/api/bowlers/${bowlerId}/workload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteWorkload: (bowlerId: string, logId: string) =>
    jsonReq<{ deleted: boolean; log_id: string }>(`/api/bowlers/${bowlerId}/workload/${logId}`, { method: "DELETE" }),
  spellAnalytics: (bowlerId: string, sessionId?: string | null) =>
    jsonReq<SpellAnalyticsResponse>(
      `/api/bowlers/${bowlerId}/spell-analytics${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`
    ),
};
