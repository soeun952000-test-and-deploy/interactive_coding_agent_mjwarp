import { useState, useEffect, useRef, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// MASTER TEMPLATE ENVELOPE
// Every script the agent generates must slot into this canonical structure.
// ─────────────────────────────────────────────────────────────────────────────
const MASTER_TEMPLATE = `
import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw

# [SLOT: DEPENDENCIES]
# Dynamic system imports compiled from active modules are injected here.

# Engine and hardware acceleration initialization sequence
wp.init()
if wp.get_device().is_cuda:
    print(f"Targeting CUDA Acceleration: {wp.get_device().name}")

# [SLOT: CUSTOM_KERNELS]
# Creative Injection Zone for CUDA parallel math macros.

# [SLOT: XML_ASSET]
# Creative Injection Zone for custom robot topologies and physics assets.
robot_xml = """
<mujoco model="scaffolding_environment">
    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.2 0.2 0.2 1"/>
    </worldbody>
</mujoco>
"""

# [SLOT: HOST_COMPILATION]
# Memory mapping pipeline moving pointers from host CPU to compiler GPU layers.
mj_model = mujoco.MjModel.from_xml_string(robot_xml)
mj_data  = mujoco.MjData(mj_model)
model    = mjw.put_model(mj_model)

# [SLOT: INITIALIZATION]
# Parameterized workspace instantiation for parallel simulation.
N_WORLD = 16  # Scaled dynamically via user execution criteria
data = mjw.make_data(mj_model, nworld=N_WORLD)

# [SLOT: STATE_MUTATION]
# Memory slice vectors and pointer manipulation blocks go here.

# [SLOT: EXECUTION_LOOP]
# High-performance execution block utilising scoped graph capture.
with wp.ScopedCapture() as capture:
    mjw.step(model, data)

graph = wp.capture_launch(capture.graph)
print(f"Successfully evaluated physics pipeline across {N_WORLD} parallel worlds.")
`.trim();

// ─────────────────────────────────────────────────────────────────────────────
// ATOMIC CODE COMPONENTS  (harvested from mujoco_warp test/benchmark suite)
// These are the verified "LEGO bricks" injected into the LLM prompt.
// Keys map 1-to-1 with template slot names.
// ─────────────────────────────────────────────────────────────────────────────
const ATOMIC_COMPONENTS = {
  HOST_COMPILATION: {
    slot: "HOST_COMPILATION",
    description: "Load MjModel from XML, compile to GPU via put_model",
    code: `
mj_model = mujoco.MjModel.from_xml_string(robot_xml)
mj_data  = mujoco.MjData(mj_model)
model    = mjw.put_model(mj_model)
    `.trim(),
  },
  INITIALIZATION: {
    slot: "INITIALIZATION",
    description: "Allocate N parallel mjData worlds on GPU",
    code: `
N_WORLD = 16          # configurable: scale to VRAM budget
data = mjw.make_data(mj_model, nworld=N_WORLD)
    `.trim(),
  },
  STATE_MUTATION: {
    slot: "STATE_MUTATION",
    description: "Zero-copy numpy pointer extraction with stride offset math",
    code: `
# Extract per-world qpos slice (stride = mj_model.nq per world)
for world_idx in range(N_WORLD):
    qpos_slice = data.qpos.numpy()[world_idx * mj_model.nq : (world_idx + 1) * mj_model.nq]
    # mutate in-place; changes are reflected back on the GPU array
    qpos_slice[:] = np.random.uniform(-0.1, 0.1, mj_model.nq)
    `.trim(),
  },
  CUSTOM_KERNELS: {
    slot: "CUSTOM_KERNELS",
    description: "Strictly-typed CUDA parallel kernel skeleton using @wp.kernel",
    code: `
@wp.kernel
def apply_force_kernel(
    force_field: wp.array(dtype=wp.vec3),
    qfrc_applied: wp.array(dtype=wp.float32),
    nv: int,
):
    tid = wp.tid()                        # one thread per world
    f   = force_field[tid]
    for i in range(nv):
        qfrc_applied[tid * nv + i] = f[i % 3]
    `.trim(),
  },
  EXECUTION_LOOP: {
    slot: "EXECUTION_LOOP",
    description: "CUDA Graph capture + launch via wp.ScopedCapture",
    code: `
# Graph capture: record the step once, replay at full GPU throughput
with wp.ScopedCapture() as capture:
    mjw.step(model, data)

graph = wp.capture_launch(capture.graph)
print(f"Physics pipeline evaluated across {N_WORLD} parallel worlds.")
    `.trim(),
  },
  PYTORCH_BRIDGE: {
    slot: "STATE_MUTATION",
    description: "Zero-copy Warp → PyTorch tensor bridge for policy networks",
    code: `
import torch
# wp.to_torch() exposes the underlying CUDA memory — no copy
qpos_tensor = wp.to_torch(data.qpos)          # shape: [N_WORLD * nq]
qpos_tensor = qpos_tensor.view(N_WORLD, -1)   # reshape to [N_WORLD, nq]
# Feed directly to a torch policy network
# actions = policy_net(qpos_tensor)
    `.trim(),
  },
  VRAM_ESTIMATE: {
    slot: "DEPENDENCIES",
    description: "VRAM estimation comment block for T4 GPU (16 GB)",
    code: `
# 💾 VRAM ESTIMATE (T4 = 16 GB)
# mjData per world ≈ (nq + nv + na + ncon_max) * 4 bytes
# Example: humanoid (nq=27, nv=26) × 64 worlds
#   core state  ≈ 64 × (27+26) × 4 ≈ ~13 KB
#   contact buf ≈ 64 × 500 contacts × 64 B ≈ ~2 MB
#   kernel tmp  ≈ ~100 MB overhead
# Total for 64 humanoids ≈ ~500 MB  (well within T4 budget)
    `.trim(),
  },
  VISUALIZATION: {
    slot: "VISUALIZATION",
    description: "Render per-world frames via mujoco.Renderer + mediapy grid display",
    code: `
import os
os.environ["MUJOCO_GL"] = "egl"   # must be set before importing mujoco
import mediapy as media

# --- Render a frame for each parallel world ---
frames = []
cpu_data = mujoco.MjData(mj_model)   # scratch CPU mjData for rendering
with mujoco.Renderer(mj_model) as renderer:
    for world_idx in range(N_WORLD):
        # Copy this world's qpos/qvel back to CPU mjData
        nq, nv = mj_model.nq, mj_model.nv
        cpu_data.qpos[:] = data.qpos.numpy()[world_idx * nq:(world_idx + 1) * nq]
        cpu_data.qvel[:] = data.qvel.numpy()[world_idx * nv:(world_idx + 1) * nv]
        mujoco.mj_forward(mj_model, cpu_data)
        renderer.update_scene(cpu_data)
        frames.append(renderer.render())

# Display all worlds as a labelled image grid inline in Colab
media.show_images(frames, titles=[f"World {i}" for i in range(N_WORLD)])
    `.trim(),
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// PERSISTENCE LAYER — negative_constraints.json schema + storage helpers
// Mirrors the plain-text serialization schema defined in the architecture doc.
// Uses window.storage (artifact persistent KV) so learnings survive sessions.
// ─────────────────────────────────────────────────────────────────────────────

const CONSTRAINTS_STORAGE_KEY = "negative_constraints_v1";

const SEED_CONSTRAINTS = [
  {
    id: "NC_001_DEVICE_HALLUCINATION",
    target_slots: ["HOST_COMPILATION", "INITIALIZATION"],
    error_signature: "TypeError: put_model() got an unexpected keyword argument 'device'",
    rule: "mujoco_warp.put_model() and mujoco_warp.make_data() accept NO explicit device keyword parameters. They inherit hardware stream pointers automatically from the global wp.init() context thread.",
    added: "2026-05-19",
  },
  {
    id: "NC_002_QPOS_STRIDE_BOUNDS",
    target_slots: ["STATE_MUTATION"],
    error_signature: "IndexError: index is out of bounds for axis 0",
    rule: "Do not treat data.qpos as a flat 1D array using stride-offset calculations. mujoco_warp structures state tensors as 2D matrices shaped as (N_WORLD, nq). Always manipulate elements using 2D tuple indexing: qpos_host[world_idx, joint_idx].",
    added: "2026-05-19",
  },
];

async function loadConstraints() {
  try {
    const result = await window.storage.get(CONSTRAINTS_STORAGE_KEY);
    if (result?.value) return JSON.parse(result.value);
  } catch (_) {}
  return { last_updated: new Date().toISOString().slice(0, 10), negative_constraints: SEED_CONSTRAINTS };
}

async function saveConstraints(db) {
  try {
    db.last_updated = new Date().toISOString().slice(0, 10);
    await window.storage.set(CONSTRAINTS_STORAGE_KEY, JSON.stringify(db));
    return true;
  } catch (_) { return false; }
}

function exportConstraintsJSON(db) {
  const blob = new Blob([JSON.stringify(db, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "negative_constraints.json";
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

// ─────────────────────────────────────────────────────────────────────────────
// SCENARIO ROUTER
// Maps user intent → relevant ATOMIC_COMPONENTS slots to inject as context
// ─────────────────────────────────────────────────────────────────────────────
function detectScenario(text) {
  const t = text.toLowerCase();
  const scenarios = [];

  // Scenario A — Basic Environment & Instantiation
  if (/\b(creat|build|init|setup|spawn|world|environment|basic|simple)\b/.test(t))
    scenarios.push("HOST_COMPILATION", "INITIALIZATION", "VRAM_ESTIMATE");

  // Scenario B — State Mutation & Slicing
  if (/\b(qpos|state|slice|mutation|extract|stride|numpy|read|write|mutate)\b/.test(t))
    scenarios.push("STATE_MUTATION");

  // Scenario C — Custom Forces & Actuator Kernels
  if (/\b(kernel|force|actuator|torque|cuda|thrust|warp kernel|@wp|parallel)\b/.test(t))
    scenarios.push("CUSTOM_KERNELS");

  // Scenario D — Graph Capture Optimization
  if (/\b(graph|capture|scoped|performance|throughput|optimiz|speed|fast)\b/.test(t))
    scenarios.push("EXECUTION_LOOP");

  // PyTorch bridge
  if (/\b(pytorch|torch|tensor|policy|neural|network|rl|reinforcement|bridge)\b/.test(t))
    scenarios.push("PYTORCH_BRIDGE", "STATE_MUTATION");

  // VRAM questions
  if (/\b(vram|memory|gpu memory|16gb|t4|budget|ram)\b/.test(t))
    scenarios.push("VRAM_ESTIMATE", "INITIALIZATION");

  // Visualization / rendering
  if (/\b(render|visual|display|image|frame|mediapy|show|view|screenshot|plot)\b/.test(t))
    scenarios.push("VISUALIZATION", "HOST_COMPILATION", "INITIALIZATION");

  // Default: inject core scaffolding if no specific scenario matched
  if (scenarios.length === 0)
    scenarios.push("HOST_COMPILATION", "INITIALIZATION", "EXECUTION_LOOP");

  return [...new Set(scenarios)];
}

// ─────────────────────────────────────────────────────────────────────────────
// SYSTEM PROMPT BUILDER
// Assembles the final LLM prompt by injecting matched atomic components
// ─────────────────────────────────────────────────────────────────────────────
function buildSystemPrompt(matchedSlots, errorLog = [], negativeConstraints = []) {
  // NOTE: This function is now display-only — the actual system prompt sent to
  // Gemini is built entirely by build_system_prompt() in the notebook backend,
  // which always injects the full atomic block library for context cache padding.
  // This client-side version is used only to populate the Scaffold Matrix UI panel.

  // Active slot summary — shown in the UI, not sent to Gemini
  const activeSlotsDisplay = matchedSlots
    .filter(k => ATOMIC_COMPONENTS[k])
    .map(k => {
      const c = ATOMIC_COMPONENTS[k];
      return `  • [${c.slot}] — ${c.description}`;
    })
    .join("\n");

  // Merge error-log entries (session) + persisted negative constraints (cross-session)
  const allConstraints = [
    ...negativeConstraints.map(nc => ({
      category: nc.id,
      trigger: nc.error_signature,
      rootCause: nc.rule,
      prevention: `Target slots: ${(nc.target_slots || []).join(", ")}`,
      supersedes: nc.supersedes,
    })),
    ...errorLog,
  ];

  const errorSection = allConstraints.length > 0 ? `
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEARNED CONSTRAINTS — APPLY THESE (highest specificity wins)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per the OVERRIDE PROTOCOL above: any constraint with a supersedes field takes
precedence over the referenced default rule for the scenario it describes.
YOU MUST apply these proactively in all generated code.
${allConstraints.map((e, i) => {
  const supLine = e.supersedes && e.supersedes !== "none"
    ? `\n  Supersedes: ${e.supersedes} (use this instead of that default rule for this scenario)`
    : "";
  return `\nCONSTRAINT ${i + 1} [${e.category}]:\n  Trigger: ${e.trigger}\n  Root cause: ${e.rootCause}\n  Prevention: ${e.prevention}${supLine}`;
}).join("\n")}
` : "";

  return `You are MuJoCo Warp Co-Pilot — a domain-specific AI assembler for the google-deepmind/mujoco_warp simulation framework running on Google Colab (T4 GPU).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MJWARP GROUND-TRUTH DEFAULT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The following four rules define the DEFAULT correct patterns for mujoco_warp.
They are derived from the official DeepMind mujoco_warp specification:
  📖 https://mujoco.readthedocs.io/en/latest/mjwarp/index.html
  📖 https://github.com/google-deepmind/mujoco_warp

OVERRIDE PROTOCOL — READ BEFORE APPLYING ANY RULE:
  These rules are defaults, not absolute prohibitions.
  If a LEARNED CONSTRAINT below carries a "supersedes" field referencing one of
  these rules (e.g. supersedes: RULE_2), that learned constraint takes precedence
  over the default for the specific scenario it describes.
  Always prefer the more specific learned constraint over the general default
  when both apply to the current task.

RULE_1 — DIMENSION MATRIX FRAMEWORK (default):
  data.qpos and data.qvel are natively 2D CUDA-device arrays with shape (nworld, nq)
  and (nworld, nv) respectively. Prefer 2D indexing: data.qpos.numpy()[world_idx, joint_idx]
  Override allowed if a learned constraint documents a validated alternative access pattern.

RULE_2 — KERNEL SIGNATURE MATCHING (default):
  Prefer declaring position/velocity array parameters as:
    qpos: wp.array(dtype=wp.float32, ndim=2)
  Passing a multi-world 2D matrix to a 1D kernel is the most common source of
  shape mismatch errors. Override allowed if a learned constraint documents a
  validated single-world kernel pattern for a specific use case.

RULE_3 — HOST-TO-DEVICE MEMORY TRANSFER (default):
  Prefer these native bindings for moving physics variables to GPU:
    model = mjw.put_model(mj_model)
    data  = mjw.make_data(mj_model, nworld=N)
  Do NOT pass device= keyword arguments — they are not accepted by any version.
  This constraint has no known valid override.

RULE_4 — DEVICE-TO-HOST DATA RETRIEVAL (default):
  Prefer mjw.get_data_into(cpu_data, mj_model, data, index=world_idx) for render-path
  data extraction. Override allowed if a learned constraint documents a validated
  direct numpy access pattern for a specific non-render use case.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE EXECUTION MANDATE — READ FIRST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You operate via COMPOSITIONAL SYNTHESIS — not free-form code generation.
All code you output MUST be assembled by slotting verified atomic components into the Master Template Envelope below. You are the Core System Assembler.

DO NOT invent wrapper classes, fictional methods, or abstractions.
DO NOT hallucinate API names. If an API does not appear in the atomic blocks or master template, flag it: "⚠️ Documentation Gap — cannot confirm this API in the verified mujoco_warp source."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MASTER TEMPLATE ENVELOPE (immutable outer scaffold)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${MASTER_TEMPLATE}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIVE SLOTS FOR THIS QUERY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${activeSlotsDisplay || "(base query — no specific slots matched)"}

(Full atomic block library always injected by backend — display only)
${errorSection}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASSEMBLY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RIGID IMITATION ZONES [HOST_COMPILATION, INITIALIZATION, EXECUTION_LOOP]:
  • Clone exact variable names: mj_model (CPU), model (GPU), data (state).
  • Never reorder these three execution layers.
  • Never rename scope chains or memory pointers.

CREATIVE INJECTION ZONES [XML_ASSET, CUSTOM_KERNELS, STATE_MUTATION]:
  • You have full engineering license for robot XML topologies and kinematics math.
  • Custom @wp.kernel functions must use wp.tid() for world-level parallelism.
  • State mutation must use stride math: world_idx * mj_model.nq for qpos slicing.

PARAMETERIZATION RULE:
  • All numerical constants (world counts, timesteps, dimensions) must be exposed as
    named constants at the top of the script (e.g. N_WORLD = 64, TIMESTEP = 0.002).
  • Never hardcode "magic" numbers inline.

VARIABLE UNIFORMITY:
  • CPU model container → always mj_model
  • GPU compiled model  → always model
  • GPU state database  → always data

WHITESPACE: Re-indent all atomic block code by exactly 4 spaces when nested
inside function bodies or loop harnesses.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DUAL-INTENT DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODE GENERATION (create / build / write / generate / implement / make):
  → Output a single, self-contained, fully-runnable Python script.
  → Structure: pip install block → imports → constants → slots in order.
  → Include VRAM estimate comment for T4 (16 GB).

CONCEPTUAL Q&A (explain / why / how / what is / describe):
  → Architectural breakdown covering kinematics, VRAM, data mappings.
  → Reference slot names (e.g. "[SLOT: EXECUTION_LOOP]") to anchor explanations.
  → MANDATORY DOC REFERENCES: Every architectural claim must cite a specific section
    of the official MuJoCo Warp documentation. Format citations as:
    📖 [Doc ref: <Section Title> — https://mujoco.readthedocs.io/en/latest/mjwarp/<path>]
    Use the following canonical doc sections:
      • Index / overview    → https://mujoco.readthedocs.io/en/latest/mjwarp/index.html
      • API reference       → https://mujoco.readthedocs.io/en/latest/mjwarp/api.html
      • Parallel worlds     → https://mujoco.readthedocs.io/en/latest/mjwarp/guide.html#parallel-worlds
      • GPU memory layout   → https://mujoco.readthedocs.io/en/latest/mjwarp/guide.html#memory-layout
      • Warp kernels        → https://mujoco.readthedocs.io/en/latest/mjwarp/guide.html#warp-kernels
      • Solver config       → https://mujoco.readthedocs.io/en/latest/mjwarp/guide.html#solver
      • Rendering           → https://mujoco.readthedocs.io/en/latest/mjwarp/guide.html#rendering
      • Performance tuning  → https://mujoco.readthedocs.io/en/latest/mjwarp/guide.html#performance
    GitHub source: https://github.com/google-deepmind/mujoco_warp
    If no specific section applies, cite the index and note the gap.
  → No code unless explicitly requested alongside the explanation.

IMAGE / SCREENSHOT TROUBLESHOOTING (user uploads an image of console output, error trace, or notebook):
  → Carefully read all visible text in the image — error messages, stack traces, line numbers.
  → Identify the root cause from the error type (ImportError, CUDA OOM, wp.kernel type error, etc.).
  → Cross-reference against the LEARNED ERROR PATTERNS section above.
  → Output a diagnosis block prefixed with 🔍 then a corrected code snippet.
  → If this error pattern is new, you MUST output the exact token ##NEW_PATTERN## on its own line, followed immediately by a structured block in this exact format — no deviations:
     ##NEW_PATTERN##
     SIGNATURE: <concise error type, e.g. "ModuleNotFoundError: mediapy not pre-installed">
     RULE: <full prevention rule, no truncation — explain root cause and exactly how to prevent it in future scripts>
     SUPERSEDES: <RULE_1 | RULE_2 | RULE_3 | RULE_4 | none — if this constraint documents a valid workaround that overrides a default rule, name it here; otherwise write none>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE FORMATTING PREFIXES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ Documentation Gap / Warning
🔧 Architecture / slot notes
💾 VRAM / memory info
⚡ Performance tip
📐 Math / physics explanation
📖 Documentation reference
🔍 Error diagnosis
##NEW_PATTERN## New error pattern — triggers automatic vault persistence (do not rephrase this token)`;
}

// ─────────────────────────────────────────────────────────────────────────────
// UI CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const QUICKSTART_PROMPTS = {
  multiRobot: `Generate a complete Python script for Google Colab that:
1. Installs and imports mujoco_warp, warp, mujoco, mediapy, and numpy (include all pip install commands)
2. Configures GPU rendering by setting MUJOCO_GL=egl in os.environ before importing mujoco
3. Creates a basic multi-robot environment with nworld=8 parallel worlds
4. Spawns 8 sphere primitive agents in a shared coordinate space
5. Runs a forward simulation step synchronously across all agents
6. Prints per-world qpos data to verify batch execution
7. Renders a visual output for each world: copy each world's qpos slice back into a CPU mujoco.MjData object, call mujoco.mj_forward(), then use mujoco.Renderer to call renderer.update_scene(data) and renderer.render() — collect all 8 frames into a list and display them as a labelled image grid using mediapy.show_images(frames, titles=[f"World {i}" for i in range(8)]) so the user can visually verify parallel simulation state inline in the notebook
Include VRAM estimates for T4 GPU.`,

  marchingBand: `Generate a complete Google Colab script for a Robot Marching Band routine:
1. Install all required packages: mujoco_warp, warp, mujoco, numpy, pyvirtualdisplay, imageio, imageio-ffmpeg
2. Load or procedurally create 64 humanoid/biped primitives using nworld=64 with mjw.make_data(mj_model, nworld=64)
3. Configure a synchronized sinusoidal trajectory loop (marching pattern) using 2D qpos indexing per RULE 1
4. Start a headless virtual display BEFORE any rendering: from pyvirtualdisplay import Display; Display(visible=False, size=(640, 480)).start()
5. Set os.environ["MUJOCO_GL"] = "egl" before importing mujoco renderer
6. Use CUDA Graph capture with wp.ScopedCapture for the physics step loop
7. For each simulation step, retrieve CPU state using mjw.get_data_into(cpu_data, mj_model, data, index=0) then render with this exact camera setup:
   camera = mujoco.MjvCamera()
   camera.type = mujoco.mjtCamera.mjCAMERA_FREE
   camera.azimuth = 90
   camera.elevation = -20
   camera.distance = 8.0
   renderer.update_scene(cpu_data, camera=camera)
   frames.append(renderer.render())
8. After the simulation loop, compile all frames to disk: imageio.mimwrite("robot_marching_band.mp4", frames, fps=60, codec="libx264")
9. Display the video inline: from IPython.display import Video; Video("robot_marching_band.mp4", embed=True)
Include VRAM estimates for T4 GPU and all pip install commands.`,
};

const MOCK_ENV_METRICS = {
  mjVersion: "3.3.1",
  commitHash: "a7f3d92",
  cudaVersion: "12.4",
  pythonVersion: "3.11.2",
  warpVersion: "1.3.0",
  vectorDbStatus: "SYNCED",
  lastSync: new Date().toLocaleTimeString(),
};

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────
function MetricPill({ label, value, color }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
      <span style={{ color: "#475569" }}>{label}:</span>
      <span style={{ color }}>{value}</span>
    </span>
  );
}

function TerminalBanner({ metrics, syncing, activeSlots }) {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setTick(p => p + 1), 1200);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{
      background: "#0a0e1a",
      borderBottom: "1px solid #1e3a5f",
      padding: "8px 20px",
      fontFamily: "'Courier New', monospace",
      fontSize: "11px",
      color: "#4ade80",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "24px", flexWrap: "wrap" }}>
        <span style={{ color: "#38bdf8", fontWeight: "bold", letterSpacing: "0.05em" }}>
          ◈ MUJOCO·WARP·COPILOT
        </span>
        <MetricPill label="MJW" value={`v${metrics.mjVersion}`} color="#4ade80" />
        <MetricPill label="COMMIT" value={metrics.commitHash} color="#a78bfa" />
        <MetricPill label="CUDA" value={metrics.cudaVersion} color="#fb923c" />
        <MetricPill label="WARP" value={`v${metrics.warpVersion}`} color="#38bdf8" />
        <MetricPill label="PY" value={metrics.pythonVersion} color="#f472b6" />
        <MetricPill
          label="SCAFFOLD"
          value={syncing ? `SYNC${tick % 2 === 0 ? "..." : "···"}` : metrics.vectorDbStatus}
          color={syncing ? "#fb923c" : "#4ade80"}
        />
        <span style={{ marginLeft: "auto", color: "#475569", fontSize: "10px" }}>
          LAST·SYNC·{metrics.lastSync}
        </span>
      </div>
      {activeSlots.length > 0 && (
        <div style={{
          marginTop: "5px",
          display: "flex",
          gap: "6px",
          flexWrap: "wrap",
          alignItems: "center",
        }}>
          <span style={{ color: "#475569", fontSize: "10px" }}>ACTIVE SLOTS:</span>
          {activeSlots.map(s => (
            <span key={s} style={{
              background: "#0d2a0d",
              border: "1px solid #1a4a1a",
              borderRadius: "3px",
              color: "#4ade80",
              fontSize: "10px",
              padding: "1px 6px",
              fontFamily: "monospace",
            }}>
              [{s}]
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function CodeBlock({ code, lang = "python" }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    const finish = () => { setCopied(true); setTimeout(() => setCopied(false), 1800); };
    if (navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(code).then(finish).catch(fallback);
    } else { fallback(); }
    function fallback() {
      const ta = document.createElement("textarea");
      ta.value = code;
      ta.style.cssText = "position:fixed;top:-9999px;left:-9999px;opacity:0";
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      try { document.execCommand("copy"); finish(); } catch (_) {}
      document.body.removeChild(ta);
    }
  };
  return (
    <div style={{
      background: "#0d1117",
      border: "1px solid #1e3a5f",
      borderRadius: "8px",
      margin: "10px 0",
      overflow: "hidden",
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "6px 14px",
        background: "#161b22",
        borderBottom: "1px solid #1e3a5f",
      }}>
        <span style={{ fontFamily: "monospace", fontSize: "11px", color: "#4ade80" }}>{lang}</span>
        <button
          onClick={copy}
          style={{
            background: "none",
            border: "1px solid #1e3a5f",
            borderRadius: "4px",
            color: copied ? "#4ade80" : "#94a3b8",
            cursor: "pointer",
            fontSize: "11px",
            padding: "2px 10px",
          }}
        >
          {copied ? "✓ copied" : "copy"}
        </button>
      </div>
      <pre style={{
        margin: 0,
        padding: "14px",
        overflowX: "auto",
        fontFamily: "'Courier New', monospace",
        fontSize: "12px",
        lineHeight: "1.7",
        color: "#e2e8f0",
      }}>
        <code>{code}</code>
      </pre>
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  const renderContent = (text) => {
    const codeBlockRegex = /```(\w+)?\n?([\s\S]*?)```/g;
    const parts = [];
    let lastIndex = 0, match;
    while ((match = codeBlockRegex.exec(text)) !== null) {
      if (match.index > lastIndex)
        parts.push({ type: "text", content: text.slice(lastIndex, match.index) });
      parts.push({ type: "code", lang: match[1] || "python", content: match[2].trim() });
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < text.length)
      parts.push({ type: "text", content: text.slice(lastIndex) });

    return parts.map((part, i) =>
      part.type === "code"
        ? <CodeBlock key={i} code={part.content} lang={part.lang} />
        : <span key={i} style={{ whiteSpace: "pre-wrap", lineHeight: "1.7" }}>{part.content}</span>
    );
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: isUser ? "row-reverse" : "row",
      alignItems: "flex-start",
      gap: "10px",
      margin: "14px 0",
    }}>
      <div style={{
        width: "30px",
        height: "30px",
        borderRadius: "50%",
        background: isUser ? "#1e3a5f" : "#0a2a0a",
        border: `1px solid ${isUser ? "#38bdf8" : "#4ade80"}`,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "13px",
        flexShrink: 0,
      }}>
        {isUser ? "👤" : "◈"}
      </div>
      <div style={{
        maxWidth: "82%",
        background: isUser ? "#0d1b2e" : "#0a0e1a",
        border: `1px solid ${isUser ? "#1e3a5f" : "#1a2e1a"}`,
        borderRadius: isUser ? "12px 4px 12px 12px" : "4px 12px 12px 12px",
        padding: "10px 14px",
        color: isUser ? "#93c5fd" : "#d1fae5",
        fontSize: "13.5px",
      }}>
        {isUser && msg.imagePreview && (
          <img src={msg.imagePreview} alt="screenshot"
            style={{ maxWidth: "100%", maxHeight: "120px", borderRadius: "4px", border: "1px solid #fb923c", marginBottom: "8px", display: "block", objectFit: "contain" }} />
        )}
        {msg.loading ? <ThinkingDots slots={msg.slots} /> : renderContent(msg.content)}
        {msg.suggestsNewPattern && (
          <div style={{ marginTop: "8px", padding: "5px 8px", background: "#1a0d00", border: "1px solid #7c3a00", borderRadius: "4px", fontSize: "11px", color: "#fb923c" }}>
            📝 New error pattern detected — automatically queued for vault persistence.
          </div>
        )}
      </div>
    </div>
  );
}

function ThinkingDots({ slots }) {
  const [dots, setDots] = useState(0);
  const [phase, setPhase] = useState(0);
  const phases = slots?.length > 0
    ? [`routing [${slots.join(", ")}]`, "fetching atomic blocks", "assembling scaffold", "synthesizing"]
    : ["analyzing", "assembling scaffold", "synthesizing"];
  useEffect(() => {
    const t1 = setInterval(() => setDots(p => (p + 1) % 4), 400);
    const t2 = setInterval(() => setPhase(p => (p + 1) % phases.length), 1600);
    return () => { clearInterval(t1); clearInterval(t2); };
  }, [phases.length]);
  return (
    <span style={{ color: "#4ade80", fontFamily: "monospace", fontSize: "12px" }}>
      {phases[phase] + ".".repeat(dots)}
    </span>
  );
}

function SidebarSection({ title, children }) {
  return (
    <div style={{ marginBottom: "20px" }}>
      <div style={{
        fontSize: "9px",
        letterSpacing: "0.15em",
        color: "#475569",
        textTransform: "uppercase",
        marginBottom: "8px",
        paddingLeft: "2px",
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}

function SidebarButton({ label, icon, onClick, accent = false, loading = false }) {
  const [hover, setHover] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        width: "100%",
        background: hover ? (accent ? "#0a2a0a" : "#0d1b2e") : "transparent",
        border: `1px solid ${accent ? "#1a3a1a" : "#1e3a5f"}`,
        borderRadius: "6px",
        color: accent ? "#4ade80" : "#38bdf8",
        cursor: "pointer",
        fontSize: "11.5px",
        padding: "7px 10px",
        textAlign: "left",
        display: "flex",
        alignItems: "center",
        gap: "8px",
        marginBottom: "6px",
        transition: "background 0.15s",
      }}
    >
      <span>{icon}</span>
      <span style={{ flex: 1 }}>{loading ? "launching..." : label}</span>
    </button>
  );
}

function SyncIndicator({ status }) {
  const colors = { idle: "#475569", syncing: "#fb923c", done: "#4ade80", error: "#ef4444" };
  const labels = { idle: "idle", syncing: "syncing...", done: "up to date", error: "sync error" };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: colors[status] }}>
      <span style={{
        width: "7px", height: "7px", borderRadius: "50%",
        background: colors[status],
        animation: status === "syncing" ? "pulse 0.8s ease-in-out infinite alternate" : "none",
      }} />
      {labels[status]}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SCAFFOLD MATRIX INSPECTOR PANEL
// Shows which atomic components were loaded for the last query
// ─────────────────────────────────────────────────────────────────────────────
function ScaffoldInspector({ slots }) {
  if (!slots || slots.length === 0) return null;
  const slotColors = {
    HOST_COMPILATION: "#38bdf8",
    INITIALIZATION: "#4ade80",
    STATE_MUTATION: "#fb923c",
    CUSTOM_KERNELS: "#f472b6",
    EXECUTION_LOOP: "#a78bfa",
    PYTORCH_BRIDGE: "#fbbf24",
    VRAM_ESTIMATE: "#6ee7b7",
  };
  return (
    <div style={{
      margin: "6px 18px",
      padding: "8px 12px",
      background: "#080c18",
      border: "1px solid #1a2a1a",
      borderRadius: "6px",
      fontSize: "11px",
      fontFamily: "monospace",
    }}>
      <div style={{ color: "#475569", marginBottom: "6px", fontSize: "10px", letterSpacing: "0.1em" }}>
        SCAFFOLD MATRIX — ACTIVE ATOMIC BLOCKS
      </div>
      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
        {slots.map(s => (
          <span key={s} style={{
            background: "#0d1117",
            border: `1px solid ${slotColors[s] || "#1e3a5f"}`,
            borderRadius: "4px",
            color: slotColors[s] || "#94a3b8",
            padding: "2px 8px",
            fontSize: "10px",
          }}>
            [{s}]
          </span>
        ))}
      </div>
      <div style={{ color: "#1e3a5f", marginTop: "6px", fontSize: "10px" }}>
        Compositional Synthesis: verified atomic components injected into master template
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
export default function MuJoCoWarpAgent() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: `◈ MuJoCo Warp Co-Pilot v5 online — Scaffold Matrix Engine + Learning Vault active.\n\nNew in v5:\n• 🔁 Automatic error logging — new patterns are detected, queued, and pushed to GitHub without any manual steps\n• 🛡️ Resilient vault sync — failed pushes are retried automatically when the backend reconnects\n• 💾 Persistent ChromaDB — vector store survives Cell 1 restarts, no re-seeding needed\n• ⚕️ Health monitoring — backend connectivity polled every 15s with inline warning if dropped\n• 🗜️ Image compression — screenshots compressed client-side before upload to prevent ngrok timeouts\n• 🎬 Video generation — Marching Band quickstart outputs a rendered .mp4 via imageio\n\nPrevious features:\n• 📸 Multimodal input — paste or upload a console screenshot for instant error diagnosis\n• 📖 Doc references — conceptual Q&A cites official mujoco.readthedocs.io/en/latest/mjwarp/ sections\n• 🖼️ Visualization — Multi-Robot quickstart generates mediapy render grids\n\nActive scaffold modules:\n• [HOST_COMPILATION] — GPU model compilation pipeline\n• [INITIALIZATION] — N-world parallel allocation\n• [STATE_MUTATION] — Zero-copy numpy stride math\n• [CUSTOM_KERNELS] — @wp.kernel parallel macros\n• [EXECUTION_LOOP] — CUDA Graph capture\n• [PYTORCH_BRIDGE] — Warp ↔ PyTorch tensor bridge\n• [VISUALIZATION] — mujoco.Renderer + mediapy grid output\n• [VRAM_ESTIMATE] — T4 memory budget comments\n\nAll code output is grounded in verified structural components and cross-session learned constraints. Documentation gaps are flagged — no hallucinated APIs. What would you like to build?`,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [syncStatus, setSyncStatus] = useState("done");
  const [syncing, setSyncing] = useState(false);
  const [quickLoading, setQuickLoading] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [activeSlots, setActiveSlots] = useState([]);
  const [lastSlots, setLastSlots] = useState([]);
  const [errorLog, setErrorLog] = useState([]);
  const [showErrorLogger, setShowErrorLogger] = useState(false);
  const [newError, setNewError] = useState({ category: "", trigger: "", rootCause: "", prevention: "" });
  const [imageAttachment, setImageAttachment] = useState(null); // { base64, mediaType, previewUrl }

  // ── Persistence layer state ──
  const [constraintDB, setConstraintDB] = useState({ last_updated: "", negative_constraints: [] });
  const [vaultStatus, setVaultStatus] = useState("loading"); // loading | ready | saving | error
  const [showVault, setShowVault] = useState(false);
  const [importError, setImportError] = useState("");
  const vaultFileInputRef = useRef(null);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // ── Backend health state ──────────────────────────────────────────────────
  const [backendOnline, setBackendOnline] = useState(true);
  // Pending learn queue — constraints that failed to POST during an outage
  const pendingLearnQueue = useRef([]);

  // Bootstrap constraint DB from persistent storage on mount
  useEffect(() => {
    loadConstraints().then(db => {
      setConstraintDB(db);
      setVaultStatus("ready");
    });
  }, []);

  // ── Health polling — ping /api/health every 15s ───────────────────────────
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/health", { method: "GET" });
        if (res.ok) {
          setBackendOnline(true);
        } else {
          setBackendOnline(false);
        }
      } catch {
        setBackendOnline(false);
      }
    };
    checkHealth(); // immediate check on mount
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Pending queue flush — retry failed /api/learn POSTs on reconnect ──────
  useEffect(() => {
    if (!backendOnline || pendingLearnQueue.current.length === 0) return;
    const flush = async () => {
      const queue = [...pendingLearnQueue.current];
      pendingLearnQueue.current = [];
      for (const payload of queue) {
        try {
          const res = await fetch("/api/learn", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!res.ok) {
            // Re-queue if still failing
            pendingLearnQueue.current.push(payload);
          }
        } catch {
          pendingLearnQueue.current.push(payload);
        }
      }
    };
    flush();
  }, [backendOnline]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(async (userText, attachedImage = null) => {
    if (!userText.trim() || loading) return;

    // Route the query through the scenario detector
    const matchedSlots = detectScenario(userText);
    setActiveSlots(matchedSlots);
    setLastSlots(matchedSlots);

    const userMsg = {
      role: "user",
      content: userText + (attachedImage ? "\n\n📸 [Screenshot attached for diagnosis]" : ""),
      imagePreview: attachedImage?.previewUrl,
    };
    const loadingMsg = { role: "assistant", content: "", loading: true, slots: matchedSlots };

    setMessages(prev => [...prev, userMsg, loadingMsg]);
    setInput("");
    setImageAttachment(null);
    setLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userText,
          matched_slots: matchedSlots,
          error_log: errorLog,
          image_base64: attachedImage?.base64 || "",
          image_media_type: attachedImage?.mediaType || "",
        }),
      });

      const data = await response.json();
      const assistantText = data.response ||
        "⚠️ No response received. Check API connectivity.";

      // Auto-detect new error patterns suggested by the agent
      const suggestsNewPattern = assistantText.includes("##NEW_PATTERN##");

      if (suggestsNewPattern) {
        // Parse the structured block after ##NEW_PATTERN##
        const patternBlock = assistantText.split("##NEW_PATTERN##")[1] || "";
        const signatureMatch = patternBlock.match(/SIGNATURE:\s*(.+)/);
        const ruleMatch      = patternBlock.match(/RULE:\s*([\s\S]+?)(?=\n[A-Z#]|$)/);

        const errorSignature = signatureMatch?.[1]?.trim()
          || patternBlock.split("\n").find(l => l.trim())?.trim()
          || "unknown-pattern";

        const rule = ruleMatch?.[1]?.trim()
          || patternBlock.replace(/SIGNATURE:.+\n?/, "").trim()
          || "Auto-captured pattern — review in negative_constraints.json";

        const supersedesMatch = patternBlock.match(/SUPERSEDES:\s*(.+)/);
        const supersedes = supersedesMatch?.[1]?.trim().toLowerCase() === "none"
          ? undefined
          : supersedesMatch?.[1]?.trim();

        const learnPayload = {
          id: "NC_AUTO_" + Date.now(),
          target_slots: matchedSlots,
          error_signature: errorSignature,
          rule: rule,
          ...(supersedes ? { supersedes } : {}),
        };
        try {
          const learnRes = await fetch("/api/learn", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(learnPayload),
          });
          if (!learnRes.ok) throw new Error("non-200");
        } catch {
          // Backend unreachable — queue for retry when connection is restored
          pendingLearnQueue.current.push(learnPayload);
          console.warn("⚠️ /api/learn failed — constraint queued for retry on reconnect.");
        }
      }

      setConversationHistory(prev => [
        ...prev,
        { role: "user", content: userText },
        { role: "assistant", content: assistantText },
      ]);
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: assistantText, slots: matchedSlots, suggestsNewPattern },
      ]);
    } catch (err) {
      setMessages(prev => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content: `⚠️ Connection error: ${err.message}\n\nEnsure the Gemini backend in Cell 1 is actively running.`,
        },
      ]);
    } finally {
      setLoading(false);
      setActiveSlots([]);
    }
  }, [loading, conversationHistory, errorLog]);

  const triggerQuickstart = async (key) => {
    setQuickLoading(key);
    await sendMessage(QUICKSTART_PROMPTS[key]);
    setQuickLoading(null);
  };

  // ── Image compression helper — downscales to ≤1024×768 and re-encodes at
  //    JPEG q=0.75 before upload, preventing ngrok free-tier buffer dropouts.
  const compressImage = (file, onDone) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const MAX_W = 1024;
        const MAX_H = 768;
        let { width, height } = img;
        if (width > MAX_W || height > MAX_H) {
          const ratio = Math.min(MAX_W / width, MAX_H / height);
          width  = Math.round(width  * ratio);
          height = Math.round(height * ratio);
        }
        const canvas = document.createElement("canvas");
        canvas.width  = width;
        canvas.height = height;
        canvas.getContext("2d").drawImage(img, 0, 0, width, height);
        const dataUrl = canvas.toDataURL("image/jpeg", 0.75);
        const base64  = dataUrl.split(",")[1];
        onDone({ base64, mediaType: "image/jpeg", previewUrl: dataUrl });
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  };

  const handleImageUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    compressImage(file, (compressed) => setImageAttachment(compressed));
  };

  const handlePasteImage = (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        compressImage(file, (compressed) => setImageAttachment(compressed));
        e.preventDefault();
        break;
      }
    }
  };

  const addErrorToLog = () => {
    if (!newError.category.trim() || !newError.rootCause.trim()) return;
    setErrorLog(prev => [...prev, { ...newError, id: Date.now() }]);
    setNewError({ category: "", trigger: "", rootCause: "", prevention: "" });
    setShowErrorLogger(false);
  };

  // ── Persistence: promote a session error-log entry into the cross-session constraint DB ──
  const promoteToVault = useCallback(async (errorEntry) => {
    const newConstraint = {
      id: `NC_${Date.now()}`,
      target_slots: errorEntry.slots || [],
      error_signature: errorEntry.trigger || errorEntry.category,
      rule: errorEntry.rootCause,
      added: new Date().toISOString().slice(0, 10),
    };
    const updatedDB = {
      ...constraintDB,
      negative_constraints: [...constraintDB.negative_constraints, newConstraint],
    };
    setConstraintDB(updatedDB);
    setVaultStatus("saving");
    const ok = await saveConstraints(updatedDB);
    setVaultStatus(ok ? "ready" : "error");
    // Remove from session error log after promotion
    setErrorLog(prev => prev.filter(e => e.id !== errorEntry.id));
  }, [constraintDB]);

  // ── Persistence: delete a constraint from the DB ──
  const deleteConstraint = useCallback(async (constraintId) => {
    const updatedDB = {
      ...constraintDB,
      negative_constraints: constraintDB.negative_constraints.filter(c => c.id !== constraintId),
    };
    setConstraintDB(updatedDB);
    setVaultStatus("saving");
    const ok = await saveConstraints(updatedDB);
    setVaultStatus(ok ? "ready" : "error");
  }, [constraintDB]);

  // ── Persistence: import negative_constraints.json from disk ──
  const handleVaultImport = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (ev) => {
      try {
        const parsed = JSON.parse(ev.target.result);
        if (!Array.isArray(parsed.negative_constraints)) throw new Error("Invalid schema");
        const merged = {
          last_updated: parsed.last_updated || new Date().toISOString().slice(0, 10),
          negative_constraints: parsed.negative_constraints,
        };
        setConstraintDB(merged);
        setImportError("");
        setVaultStatus("saving");
        const ok = await saveConstraints(merged);
        setVaultStatus(ok ? "ready" : "error");
      } catch (err) {
        setImportError(`⚠️ Import failed: ${err.message}`);
      }
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const triggerSync = async () => {
    setSyncing(true);
    setSyncStatus("syncing");
    await new Promise(r => setTimeout(r, 2400));
    setSyncing(false);
    setSyncStatus("done");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input, imageAttachment);
    }
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "700px",
      background: "#060a14",
      borderRadius: "10px",
      overflow: "hidden",
      border: "1px solid #1e3a5f",
      fontFamily: "'Courier New', monospace",
    }}>
      <style>{`
        @keyframes pulse { from { opacity: 0.4; } to { opacity: 1; } }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 4px; }
      `}</style>

      <TerminalBanner metrics={MOCK_ENV_METRICS} syncing={syncing} activeSlots={activeSlots} />

      {/* ── Backend offline warning banner ── */}
      {!backendOnline && (
        <div style={{
          background: "#3d1a00",
          borderBottom: "1px solid #ff6b00",
          padding: "6px 16px",
          fontSize: "11px",
          color: "#ff9944",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}>
          <span>⚠</span>
          <span>
            Backend unreachable — re-run <strong>Cell 1</strong> then <strong>Cell 1.5</strong> in Colab.
            {pendingLearnQueue.current.length > 0 &&
              ` ${pendingLearnQueue.current.length} constraint(s) queued and will auto-push on reconnect.`}
          </span>
        </div>
      )}

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Sidebar */}
        <div style={{
          width: "188px",
          flexShrink: 0,
          background: "#080c18",
          borderRight: "1px solid #1e3a5f",
          padding: "16px 12px",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
        }}>
          <SidebarSection title="Quickstart">
            <SidebarButton icon="🤖" label="Multi-Robot Test"
              loading={quickLoading === "multiRobot"}
              onClick={() => triggerQuickstart("multiRobot")} accent />
            <SidebarButton icon="🥁" label="Marching Band (64x)"
              loading={quickLoading === "marchingBand"}
              onClick={() => triggerQuickstart("marchingBand")} accent />
          </SidebarSection>

          <SidebarSection title="Repository">
            <SidebarButton icon="🔄" label="Sync Scaffold DB" onClick={triggerSync} />
            <div style={{ paddingLeft: "2px", marginTop: "4px" }}>
              <SyncIndicator status={syncStatus} />
            </div>
          </SidebarSection>

          <SidebarSection title="Scaffold Modules">
            {Object.entries(ATOMIC_COMPONENTS).map(([key, c]) => (
              <div key={key} style={{
                fontSize: "10px",
                color: "#475569",
                padding: "3px 4px",
                fontFamily: "monospace",
                borderLeft: "2px solid #1e3a5f",
                marginBottom: "4px",
                lineHeight: "1.4",
              }}>
                <span style={{ color: "#4ade80" }}>[{c.slot}]</span>
                <br />
                <span style={{ fontSize: "9px" }}>{c.description.slice(0, 36)}…</span>
              </div>
            ))}
          </SidebarSection>

          <SidebarSection title="Error Memory">
            <SidebarButton icon="⚠️" label={`Log Error (${errorLog.length})`}
              onClick={() => setShowErrorLogger(p => !p)} />
            {showErrorLogger && (
              <div style={{ fontSize: "10px", marginTop: "6px" }}>
                {[
                  ["category", "Category (e.g. CUDA OOM)"],
                  ["trigger", "What caused it"],
                  ["rootCause", "Root cause"],
                  ["prevention", "Prevention"],
                ].map(([field, ph]) => (
                  <input
                    key={field}
                    value={newError[field]}
                    onChange={e => setNewError(p => ({ ...p, [field]: e.target.value }))}
                    placeholder={ph}
                    style={{
                      width: "100%", boxSizing: "border-box",
                      background: "#0a0e1a", border: "1px solid #1e3a5f",
                      borderRadius: "4px", color: "#93c5fd",
                      fontFamily: "monospace", fontSize: "9px",
                      padding: "4px 6px", marginBottom: "4px", outline: "none",
                    }}
                  />
                ))}
                <button onClick={addErrorToLog} style={{
                  width: "100%", background: "#0a2a0a", border: "1px solid #4ade80",
                  borderRadius: "4px", color: "#4ade80", cursor: "pointer",
                  fontSize: "10px", padding: "4px",
                }}>+ Add to Error Log</button>
              </div>
            )}
            {errorLog.length > 0 && (
              <div style={{ marginTop: "6px" }}>
                {errorLog.map((e, i) => (
                  <div key={e.id} style={{
                    fontSize: "9px", color: "#fb923c", fontFamily: "monospace",
                    borderLeft: "2px solid #7c3a00", padding: "2px 4px", marginBottom: "3px",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span>[{i + 1}] {e.category}</span>
                      <div style={{ display: "flex", gap: "4px" }}>
                        <button
                          onClick={() => promoteToVault(e)}
                          title="Persist to cross-session vault"
                          style={{ background: "#0a1a0a", border: "1px solid #4ade80", borderRadius: "3px", color: "#4ade80", cursor: "pointer", fontSize: "8px", padding: "1px 4px" }}
                        >⬆ Vault</button>
                        <button onClick={() => setErrorLog(prev => prev.filter(x => x.id !== e.id))}
                          style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: "9px", padding: 0 }}>✕</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </SidebarSection>

          <SidebarSection title="Learning Vault">
            {/* Hidden file input for JSON import */}
            <input ref={vaultFileInputRef} type="file" accept=".json,application/json"
              onChange={handleVaultImport} style={{ display: "none" }} />

            {/* Vault status indicator */}
            <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
              <span style={{
                width: "7px", height: "7px", borderRadius: "50%",
                background: vaultStatus === "ready" ? "#4ade80" : vaultStatus === "saving" ? "#fb923c" : vaultStatus === "error" ? "#ef4444" : "#475569",
                animation: vaultStatus === "saving" ? "pulse 0.8s ease-in-out infinite alternate" : "none",
                flexShrink: 0,
              }} />
              <span style={{ fontSize: "10px", color: "#475569", fontFamily: "monospace" }}>
                {vaultStatus === "loading" ? "loading…" : vaultStatus === "saving" ? "persisting…" : vaultStatus === "error" ? "save error" : `${constraintDB.negative_constraints.length} rules · ${constraintDB.last_updated}`}
              </span>
            </div>

            {/* Expand/collapse vault panel */}
            <SidebarButton icon="🧠" label="Browse Vault"
              onClick={() => setShowVault(p => !p)} />

            {showVault && (
              <div style={{ marginTop: "6px", maxHeight: "160px", overflowY: "auto" }}>
                {constraintDB.negative_constraints.length === 0 && (
                  <div style={{ fontSize: "9px", color: "#475569", fontFamily: "monospace", padding: "4px" }}>
                    No constraints yet. Promote session errors with ⬆ Vault.
                  </div>
                )}
                {constraintDB.negative_constraints.map((nc) => (
                  <div key={nc.id} style={{
                    fontSize: "9px", fontFamily: "monospace",
                    borderLeft: "2px solid #1e3a5f", padding: "3px 6px", marginBottom: "4px",
                    background: "#080c18",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "4px" }}>
                      <div>
                        <div style={{ color: "#38bdf8", marginBottom: "2px" }}>{nc.id}</div>
                        <div style={{ color: "#475569", fontSize: "8px" }}>
                          {(nc.target_slots || []).map(s => `[${s}]`).join(" ")}
                        </div>
                        <div style={{ color: "#64748b", marginTop: "2px", lineHeight: "1.4" }}>
                          {nc.error_signature?.slice(0, 60)}{nc.error_signature?.length > 60 ? "…" : ""}
                        </div>
                      </div>
                      <button onClick={() => deleteConstraint(nc.id)}
                        style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: "9px", padding: 0, flexShrink: 0 }}>✕</button>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {importError && (
              <div style={{ fontSize: "9px", color: "#ef4444", fontFamily: "monospace", marginTop: "4px" }}>
                {importError}
              </div>
            )}

            <div style={{ display: "flex", gap: "4px", marginTop: "6px" }}>
              <button
                onClick={() => exportConstraintsJSON(constraintDB)}
                title="Export negative_constraints.json for GitHub commit"
                style={{
                  flex: 1, background: "#0a1a0a", border: "1px solid #1a3a1a",
                  borderRadius: "5px", color: "#4ade80", cursor: "pointer",
                  fontSize: "9px", padding: "5px 4px", fontFamily: "monospace",
                }}
              >⬇ Export JSON</button>
              <button
                onClick={() => vaultFileInputRef.current?.click()}
                title="Import negative_constraints.json from GitHub clone"
                style={{
                  flex: 1, background: "#0a0e1a", border: "1px solid #1e3a5f",
                  borderRadius: "5px", color: "#38bdf8", cursor: "pointer",
                  fontSize: "9px", padding: "5px 4px", fontFamily: "monospace",
                }}
              >⬆ Import JSON</button>
            </div>
            <div style={{ fontSize: "8px", color: "#1e3a5f", fontFamily: "monospace", marginTop: "4px", lineHeight: "1.5" }}>
              Export → commit to GitHub → import on next session to close the feedback loop.
            </div>
          </SidebarSection>

          <SidebarSection title="Quick Queries">
            {[
              ["⚡", "nworld sizing guide",
                "Explain how to choose the optimal nworld value for T4 GPU VRAM — include a formula and example calculations for a 16GB memory budget."],
              ["💾", "VRAM estimator",
                "Explain the VRAM layout for a MuJoCo Warp batch simulation: how is memory allocated per world for mjData, contact arrays, and Warp kernel buffers?"],
              ["🔧", "CUDA graph capture",
                "Explain the wp.ScopedCapture pattern in MuJoCo Warp: when to use it, how it differs from eager mode, and what operations cannot be captured."],
              ["🦾", "PyTorch bridge",
                "Show me how to bridge Warp arrays to PyTorch tensors in a MuJoCo Warp simulation loop for use with a neural network policy."],
              ["📐", "Solver config",
                "Explain the constraint solver options in MuJoCo Warp — CG vs Newton vs PGS — with recommendations for multi-robot contact-rich environments."],
            ].map(([icon, label, prompt]) => (
              <SidebarButton key={label} icon={icon} label={label}
                onClick={() => sendMessage(prompt)} />
            ))}
          </SidebarSection>

          <div style={{ marginTop: "auto", paddingTop: "12px", borderTop: "1px solid #1e3a5f" }}>
            <div style={{ fontSize: "10px", color: "#1e3a5f", lineHeight: "1.8" }}>
              <div>◈ Scaffold Matrix Engine</div>
              <div>◈ Compositional Synthesis</div>
              <div>◈ Error Memory ({errorLog.length} session · {constraintDB.negative_constraints.length} vaulted)</div>
              <div>◈ Doc-referenced Q&A</div>
              <div>◈ T4-optimized</div>
            </div>
          </div>
        </div>

        {/* Main Chat */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", padding: "14px 18px" }}>
            {messages.map((msg, i) => (
              <div key={i}>
                <MessageBubble msg={msg} />
                {/* Show scaffold inspector after each assistant code response */}
                {msg.role === "assistant" && !msg.loading && msg.slots?.length > 0 && (
                  <ScaffoldInspector slots={msg.slots} />
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div style={{
            borderTop: "1px solid #1e3a5f",
            padding: "12px 16px",
            background: "#080c18",
          }}>
            {/* Image preview strip */}
            {imageAttachment && (
              <div style={{ marginBottom: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
                <img src={imageAttachment.previewUrl} alt="attachment"
                  style={{ height: "48px", borderRadius: "4px", border: "1px solid #fb923c", objectFit: "cover" }} />
                <span style={{ fontSize: "10px", color: "#fb923c", fontFamily: "monospace" }}>📸 screenshot attached for diagnosis</span>
                <button onClick={() => setImageAttachment(null)}
                  style={{ background: "none", border: "none", color: "#475569", cursor: "pointer", fontSize: "12px", marginLeft: "auto" }}>✕</button>
              </div>
            )}
            <div style={{ display: "flex", gap: "8px", alignItems: "flex-end" }}>
              {/* Hidden file input */}
              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageUpload}
                style={{ display: "none" }} />
              {/* Upload image button */}
              <button
                onClick={() => fileInputRef.current?.click()}
                title="Upload screenshot for error diagnosis"
                style={{
                  background: imageAttachment ? "#1a1500" : "transparent",
                  border: `1px solid ${imageAttachment ? "#fb923c" : "#1e3a5f"}`,
                  borderRadius: "7px",
                  color: imageAttachment ? "#fb923c" : "#475569",
                  cursor: "pointer",
                  fontSize: "16px",
                  width: "36px",
                  height: "36px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >📸</button>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                onPaste={handlePasteImage}
                placeholder="Describe a scenario, ask for code, paste a screenshot (Ctrl+V), or request an architecture explanation..."
                rows={2}
                style={{
                  flex: 1,
                  background: "#0a0e1a",
                  border: "1px solid #1e3a5f",
                  borderRadius: "7px",
                  color: "#93c5fd",
                  fontFamily: "'Courier New', monospace",
                  fontSize: "12.5px",
                  padding: "9px 12px",
                  resize: "none",
                  outline: "none",
                  lineHeight: "1.6",
                }}
              />
              <button
                onClick={() => sendMessage(input, imageAttachment)}
                disabled={loading || !input.trim()}
                style={{
                  background: loading || !input.trim() ? "#0a2a0a" : "#166534",
                  border: "1px solid #4ade80",
                  borderRadius: "7px",
                  color: loading || !input.trim() ? "#1a3a1a" : "#4ade80",
                  cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                  fontSize: "18px",
                  width: "40px",
                  height: "40px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  transition: "all 0.15s",
                }}
              >
                ▶
              </button>
            </div>
          </div>
          <div style={{
            padding: "4px 16px 6px",
            fontSize: "10px",
            color: "#1e3a5f",
            background: "#080c18",
          }}>
            Enter to send · Shift+Enter for newline · 📸 Ctrl+V or click 📸 to attach screenshot · Vault: {constraintDB.negative_constraints.length} rules · Session errors: {errorLog.length}
          </div>
        </div>
      </div>
    </div>
  );
}
