import type { DeliveryDetailResponse, SessionSummary } from "./types";

const BASE = "";

async function jsonReq<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, init);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`${res.status}: ${txt}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => jsonReq<{ status: string; version: string; gpu_worker: any; mcc_landmark_keys: string[] }>("/api/health"),
  listSessions: () => jsonReq<{ sessions: SessionSummary[] }>("/api/sessions"),
  getSession: (id: string) => jsonReq<{ session: SessionSummary }>(`/api/sessions/${id}`),
  createSession: (body: { bowler_name: string; bowling_arm: string; age_band: string; title: string; notes?: string }) =>
    jsonReq<{ session: SessionSummary }>("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  getDelivery: (id: string) => jsonReq<DeliveryDetailResponse>(`/api/deliveries/${id}`),
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
  uploadDelivery: (sessionId: string, fd: FormData) =>
    jsonReq<{ delivery_id: string; recompute_ms: number; fact_sheet: DeliveryDetailResponse["fact_sheet"] }>(
      `/api/sessions/${sessionId}/deliveries/upload`,
      { method: "POST", body: fd }
    ),
};
