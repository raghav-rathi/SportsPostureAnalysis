import React, { useState } from "react";

type SetupMode = "side" | "rear" | "dual";

const MATRIX_ROWS = [
  { metric: "Front Knee Angle & Brace (FFC → BR)", side: "✓ High", rear: "Withheld", dual: "✓ High" },
  { metric: "Forward Trunk Flexion (FFC → BR)", side: "✓ High", rear: "Withheld", dual: "✓ High" },
  { metric: "Thorax Lateral Flexion (Side Bend)", side: "Withheld", rear: "✓ High", dual: "✓ High" },
  { metric: "Shoulder Counter-Rotation (SCR)", side: "Moderate", rear: "✓ High", dual: "✓ High" },
  { metric: "Pelvic Drop at Front-Foot Contact", side: "Withheld", rear: "✓ High", dual: "✓ High" },
  { metric: "Run-Up Speed & Delivery Stride (m/s)", side: "✓ Static + Crease", rear: "Withheld", dual: "✓ High" },
  { metric: "Optical Ball Release Speed (km/h)", side: "✓ ≥120 fps + Crease", rear: "Withheld", dual: "✓ High" },
  { metric: "ICC 15° Elbow Extension Legality", side: "REFUSED", rear: "REFUSED", dual: "REFUSED" },
];

export const FilmingGuideView: React.FC<{ onOpenUpload: () => void }> = ({ onOpenUpload }) => {
  const [setup, setSetup] = useState<SetupMode>("side");

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="flex flex-wrap items-end justify-between gap-4 pb-4 border-b border-[#ebebeb]">
        <div>
          <div className="v-eyebrow">▲ 30_SECOND_FILMING_PROTOCOL</div>
          <h1 className="text-[30px] font-[450] tracking-[-0.05em] text-[#171717] mt-1">
            How to Film for Lab-Grade Accuracy
          </h1>
          <p className="text-[14px] text-[#4d4d4d] mt-1 max-w-2xl">
            Every biomechanical metric has a camera-geometry requirement. Film from the right angle on a locked tripod
            at 120–240 fps so the engine unlocks high-confidence intervals instead of withholding noisy estimates.
          </p>
        </div>
        <button onClick={onOpenUpload} className="v-btn-primary h-9 px-4 text-[13px]">
          ▲ Upload Your Video
        </button>
      </div>

      {/* Interactive Setup Switcher + Top-Down Pitch Diagram */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div className="lg:col-span-7 v-card p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="v-eyebrow">CAMERA PLACEMENT DIAGRAM</div>
            <div className="flex rounded-[6px] bg-[#f4f4f5] p-0.5 font-mono text-[11px]">
              {(
                [
                  ["side", "Side-On 90°"],
                  ["rear", "Rear 0°"],
                  ["dual", "Dual-Phone 3D"],
                ] as Array<[SetupMode, string]>
              ).map(([id, lbl]) => (
                <button
                  key={id}
                  onClick={() => setSetup(id)}
                  className={`px-3 py-1 rounded-[5px] transition ${
                    setup === id ? "bg-[#171717] text-[#ffffff]" : "text-[#4d4d4d] hover:text-[#171717]"
                  }`}
                >
                  {lbl}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-[6px] bg-[#fafafa] border border-[#ebebeb] p-4">
            <svg viewBox="0 0 540 240" className="w-full h-auto">
              {/* Pitch Strip */}
              <rect x="70" y="75" width="400" height="90" rx="2" fill="#f0f0f0" stroke="#c9c9c9" />
              {/* Bowler Crease */}
              <line x1="120" y1="88" x2="120" y2="152" stroke="#171717" strokeWidth="1.5" />
              <line x1="158" y1="78" x2="158" y2="162" stroke="#297a3a" strokeWidth="2.2" />
              <line x1="100" y1="88" x2="158" y2="88" stroke="#8f8f8f" strokeWidth="1.2" />
              <line x1="100" y1="152" x2="158" y2="152" stroke="#8f8f8f" strokeWidth="1.2" />
              {/* Striker Crease */}
              <line x1="420" y1="88" x2="420" y2="152" stroke="#171717" strokeWidth="1.5" />
              <line x1="382" y1="78" x2="382" y2="162" stroke="#8f8f8f" strokeWidth="1.5" />
              {/* Stumps */}
              {[112, 120, 128].map((y, i) => (
                <circle key={`bs-${i}`} cx="120" cy={y} r="2.2" fill="#171717" />
              ))}
              {[112, 120, 128].map((y, i) => (
                <circle key={`ss-${i}`} cx="420" cy={y} r="2.2" fill="#171717" />
              ))}
              {/* Bowler Run-up Arrow */}
              <line x1="78" y1="120" x2="148" y2="120" stroke="#171717" strokeWidth="1.5" strokeDasharray="4 3" />
              <polygon points="154,120 146,116 146,124" fill="#171717" />
              <text x="158" y="68" textAnchor="middle" fill="#297a3a" fontSize="10" fontFamily="monospace">
                POPPING CREASE
              </text>
              <text x="270" y="124" textAnchor="middle" fill="#8f8f8f" fontSize="10" fontFamily="monospace">
                20.12 m PITCH LENGTH
              </text>

              {/* Side-On Camera */}
              {(setup === "side" || setup === "dual") && (
                <g>
                  <line x1="158" y1="208" x2="158" y2="128" stroke="#297a3a" strokeWidth="1.4" strokeDasharray="5 4" />
                  <rect x="138" y="204" width="40" height="22" rx="4" fill="#171717" />
                  <text x="158" y="218" textAnchor="middle" fill="#ffffff" fontSize="9" fontFamily="monospace">
                    CAM A
                  </text>
                  <text x="210" y="219" fill="#171717" fontSize="10" fontFamily="monospace">
                    Side-On 90° · 8–12 m perpendicular, hip height
                  </text>
                </g>
              )}

              {/* Rear Camera */}
              {(setup === "rear" || setup === "dual") && (
                <g>
                  <line x1="30" y1="120" x2="115" y2="120" stroke="#297a3a" strokeWidth="1.4" strokeDasharray="5 4" />
                  <rect x="10" y="109" width="38" height="22" rx="4" fill="#171717" />
                  <text x="29" y="123" textAnchor="middle" fill="#ffffff" fontSize="9" fontFamily="monospace">
                    {setup === "dual" ? "CAM B" : "CAM"}
                  </text>
                  <text x="14" y="54" fill="#171717" fontSize="10" fontFamily="monospace">
                    Rear 0° · 12–25 m behind bowler stumps (both wickets visible!)
                  </text>
                </g>
              )}
            </svg>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[12px]">
            <div className="v-card-subtle p-3">
              <div className="v-eyebrow text-[#171717]">1. LOCK THE TRIPOD</div>
              <p className="text-[#4d4d4d] mt-1">
                Do not pan or zoom during the delivery. Keep the popping crease and full body (feet to raised hand) in frame.
              </p>
            </div>
            <div className="v-card-subtle p-3">
              <div className="v-eyebrow text-[#171717]">2. USE 120–240 FPS</div>
              <p className="text-[#4d4d4d] mt-1">
                iPhone: <strong>Slo-Mo (120/240 fps)</strong>. Android: <strong>Pro Video 120 fps</strong> or Slo-Mo. Avoid WhatsApp compression.
              </p>
            </div>
            <div className="v-card-subtle p-3">
              <div className="v-eyebrow text-[#171717]">3. SHOW THE CREASE</div>
              <p className="text-[#4d4d4d] mt-1">
                Include the bowling &amp; popping creases (or both sets of wickets from behind) so real metric scale unlocks automatically.
              </p>
            </div>
          </div>
        </div>

        {/* Right: Metric Unlock Matrix */}
        <div className="lg:col-span-5 v-card p-5 space-y-3">
          <div>
            <div className="v-eyebrow">METRIC_UNLOCK_MATRIX</div>
            <h2 className="text-[16px] font-[500] text-[#171717] mt-0.5">
              What Each Camera Angle Unlocks
            </h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-[12px]">
              <thead>
                <tr className="border-b border-[#ebebeb] font-mono text-[10px] uppercase tracking-[0.06em] text-[#666666]">
                  <th className="py-2 pr-2">Metric</th>
                  <th className="py-2 px-1.5">Side 90°</th>
                  <th className="py-2 px-1.5">Rear 0°</th>
                  <th className="py-2 pl-1.5">Dual 3D</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#ebebeb]">
                {MATRIX_ROWS.map((row) => (
                  <tr key={row.metric}>
                    <td className="py-2 pr-2 text-[#171717] font-[450]">{row.metric}</td>
                    {[row.side, row.rear, row.dual].map((cell, idx) => (
                      <td
                        key={idx}
                        className={`py-2 px-1.5 font-mono text-[11px] whitespace-nowrap ${
                          cell.startsWith("✓")
                            ? "text-[#297a3a] font-medium"
                            : cell === "REFUSED"
                            ? "text-[#171717] font-medium"
                            : "text-[#8f8f8f]"
                        }`}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="p-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] text-[12px] text-[#4d4d4d]">
            <strong className="text-[#171717]">Dual-Phone 3D Sync:</strong> No hardware genlock needed. Just start recording
            on two phones (one Side-On, one Rear), upload both into the same session, and click{" "}
            <strong>Fuse 2 Angles</strong> in <em>Compare &amp; Spell</em> — the engine aligns them to &lt;2.5 ms automatically.
          </div>
        </div>
      </div>
    </div>
  );
};
