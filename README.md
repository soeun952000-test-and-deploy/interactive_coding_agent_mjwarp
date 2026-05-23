# ◈ MuJoCo Warp Co-Pilot

A domain-specific AI coding agent for the [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) physics simulation framework. Built to run entirely on a free Google Colab T4 GPU, it generates production-grade parallel simulation code, diagnoses runtime errors from screenshots, and gets smarter across sessions by persisting learned constraints to a GitHub vault.

---

## What it does

Bridging the steep barrier to entry in GPU-accelerated robotics simulation is the core goal. Rather than wrapping a general-purpose LLM around documentation, this agent uses a **Compositional Synthesis** architecture — verified atomic code components are assembled into a rigid master template, so the model never has to guess at API boundaries. If a requested pattern isn't in the verified reference library, it says so explicitly instead of hallucinating syntax.

The agent handles two intent types:

**Code generation** — transforms natural language into self-contained, immediately runnable Python scripts targeting the mujoco_warp API. Output is always structured around the Master Template Envelope with atomic blocks slotted into their correct positions.

**Conceptual Q&A** — provides architectural breakdowns of physical kinematics, VRAM allocation math, multi-robot data mappings, and API design decisions, always citing the official documentation at [mujoco.readthedocs.io/en/latest/mjwarp](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html).

---

## Architecture

```
[ Web Browser ]
      |
 (ngrok tunnel)
      ↓
[ Google Colab — Free T4 GPU ]
  ├── Cell 1   : FastAPI backend + Gemini 3.1 Flash Lite + Scaffold Matrix engine
  ├── Cell 1.5 : Cross-session learning vault (ChromaDB + GitHub sync)
  ├── Cell 2   : React frontend (Vite dev server)
  └── Cell 3   : Emergency reset utility
```

### Scaffold Matrix Engine

Every generated script is assembled from **Atomic Components** — small, verified code blocks harvested from the mujoco_warp test suite and benchmarks. These map to named slots in a universal Master Template:

| Slot | Role | Zone |
|------|------|------|
| `[HOST_COMPILATION]` | CPU → GPU model compilation via `mjw.put_model()` | Rigid imitation |
| `[INITIALIZATION]` | N-world parallel allocation via `mjw.make_data()` | Rigid imitation |
| `[EXECUTION_LOOP]` | CUDA Graph capture via `wp.ScopedCapture` | Rigid imitation |
| `[XML_ASSET]` | Custom robot topology definition | Creative injection |
| `[CUSTOM_KERNELS]` | `@wp.kernel` parallel CUDA math | Creative injection |
| `[STATE_MUTATION]` | Zero-copy numpy stride operations | Creative injection |
| `[PYTORCH_BRIDGE]` | Warp ↔ PyTorch tensor bridge | Creative injection |
| `[VISUALIZATION]` | `mujoco.Renderer` + mediapy grid output | Creative injection |
| `[VRAM_ESTIMATE]` | T4 memory budget comments | Reference |

Rigid imitation zones enforce exact variable names and scope chains. Creative injection zones give the model full engineering license for robot XML, kinematics math, and kernel design.

### Ground-Truth Default Rules

Four verified patterns from the official mujoco_warp specification are always injected into the system prompt as defaults. These are not hard prohibitions — the override protocol allows learned constraints to supersede any default for a specific validated scenario:

- **RULE_1** — `data.qpos` and `data.qvel` are 2D arrays shaped `(nworld, nq)` / `(nworld, nv)`
- **RULE_2** — `@wp.kernel` position/velocity parameters must declare `ndim=2`
- **RULE_3** — Host-to-device transfer uses `mjw.put_model()` and `mjw.make_data()` — no `device=` keyword
- **RULE_4** — Device-to-host retrieval uses `mjw.get_data_into(cpu_data, mj_model, data, index=i)`

### Self-Correction Learning Vault

When the agent identifies a new error pattern it hasn't seen before, it emits a structured `##NEW_PATTERN##` block containing a `SIGNATURE`, `RULE`, and optional `SUPERSEDES` field. The frontend intercepts this, POSTs the constraint to `/api/learn`, which appends it to `project_files/negative_constraints.json` and commits it back to GitHub automatically. On the next session, Cell 1.5 re-seeds ChromaDB from the updated file, so every learned fix carries forward.

Constraints with a `supersedes` field explicitly override the referenced default rule for their specific scenario — allowing the vault to accumulate nuanced, context-specific knowledge rather than fighting against the hardcoded defaults.

### Context Caching

The full system prompt — ground-truth rules, complete atomic block library, and vault constraints — is uploaded to Gemini's server-side context cache on the first request of each session. Subsequent requests send only the user message, reducing billable input tokens by approximately 70% per call. The cache invalidates automatically whenever a new constraint is learned.

---

## Key features

**Multimodal debugging** — paste or drag a Colab traceback screenshot directly into the chat. The agent reads the error, maps it to the broken scaffold slot, and outputs a corrected script without rewriting surrounding code.

**Image compression** — screenshots are compressed client-side to ≤1024×768 at JPEG q=0.75 before crossing the ngrok tunnel, preventing free-tier buffer timeouts on large images.

**Health monitoring** — the frontend polls `/api/health` every 15 seconds. If the Colab backend goes idle, an inline warning banner appears with instructions to re-run the relevant cells.

**Pending queue** — if a `/api/learn` POST fails during a backend outage, the constraint is held in a client-side queue and retried automatically when connectivity is restored.

**Sandboxed execution** — the `/api/execute` endpoint runs generated scripts in a subprocess with full `stdout`/`stderr` capture and timeout handling, returning structured output for the self-healing loop.

**Quickstart prompts** — one-click sidebar buttons for common tasks:
- **Multi-Robot Test** — baseline parallel simulation diagnostic
- **Marching Band** — 64 humanoid primitives in synchronized trajectory, rendered to `robot_marching_band.mp4` via `imageio` and displayed inline

---

## Setup

### Prerequisites

- Google account with Colab access
- Gemini API key — [aistudio.google.com](https://aistudio.google.com)
- GitHub account + personal access token with `Contents: Read and Write` scope
- ngrok account + authtoken — [ngrok.com](https://ngrok.com)

### Colab Secrets

Before running any cells, add these five secrets in the Colab sidebar (🔑 icon):

| Secret name | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key |
| `GITHUB_TOKEN` | Your GitHub PAT |
| `GITHUB_USERNAME` | Your GitHub username |
| `GITHUB_REPO_NAME` | This repo's name |
| `NGROK_AUTHTOKEN` | Your ngrok authtoken |

### GitHub repo setup

Create `project_files/negative_constraints.json` in your repo with this seed content:

```json
{
    "last_updated": "2026-05-21",
    "negative_constraints": [],
    "atomic_blocks": {}
}
```

### Running

1. Open `mujoco_warp_copilot_v5.ipynb` in Colab — set runtime to **T4 GPU**
2. Upload `mujoco_warp_agent_v5.jsx` to `/content/` in the Colab file tree
3. Run cells in order: **Cell 1 → Cell 1.5 → Cell 2**
4. Cell 2 prints a live ngrok URL — open it in your browser

When Cell 1 completes you should see:
```
✅ BACKEND LIVE: Scaffold Matrix engine + Slot Validator + Gemini gemini-3.1-flash-lite on port 8000!
```

When Cell 2 completes, the React UI is live at the ngrok URL.

### Mid-session constraint reload

If you want to apply newly learned constraints without restarting:

```
Re-run Cell 1.5 only
```

This re-clones the repo, reloads the updated `negative_constraints.json`, and rebuilds ChromaDB. Cell 1 and Cell 2 do not need to be re-run.

---

## Project file structure

```
/
├── mujoco_warp_copilot_v5.ipynb   # Backend notebook (FastAPI + Gemini)
├── mujoco_warp_agent_v5.jsx       # Frontend React UI
├── project_files/
│   ├── negative_constraints.json  # Cross-session learning vault
│   └── chroma_db/                 # Persisted ChromaDB vector store (auto-created)
└── README.md
```

---

## API reference

The FastAPI backend exposes four endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST | Main inference — accepts message, matched slots, error log, optional image |
| `/api/learn` | POST | Persist a new constraint to ChromaDB + GitHub |
| `/api/execute` | POST | Run a generated Python script in a sandboxed subprocess |
| `/api/health` | GET | Heartbeat — returns status, ChromaDB rule count, timestamp |

---

## Notes on API quotas

This project uses the Gemini API. On the free tier, `gemini-3.1-flash-lite` allows a limited number of requests per day. Context caching significantly reduces per-request token consumption but requires billing to be enabled. Enabling billing on your Google AI Studio project removes the hard daily cap at a cost of fractions of a cent per request.

---

## Known limitations

**Context caching requires a billing-enabled Gemini account**
Gemini's context caching feature is not available on the free tier — the storage quota is set to 0 regardless of model. The agent falls back to inline prompt injection automatically, so functionality is unaffected, but token consumption per request will be higher. Enabling billing on your Google AI Studio project unlocks caching immediately.

**Screenshot analysis may not be available on the free tier**
Multimodal (image) inference is subject to the same free-tier quota constraints as text requests. If you hit the daily limit, image uploads will return a "No response received" error. The workaround is to paste the traceback as text directly into the chat instead, which consumes fewer tokens and works reliably within free-tier limits.

**The agent may repeat known errors on edge cases**
MuJoCo Warp is a relatively new framework with significantly less training data available compared to the broader MuJoCo ecosystem. On edge cases — particularly unusual kernel configurations, non-standard nworld layouts, or newer API surface — the agent may fall back on general MuJoCo patterns that don't map correctly to the mujoco_warp runtime. This can result in the same error appearing across multiple generation attempts even after logging. The recommended workaround is to promote the pattern to the learning vault, re-run Cell 1.5 to reload constraints into ChromaDB, and include the specific error signature explicitly in your follow-up prompt so the constraint is in active context.

---

## References

- [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html)
- [google-deepmind/mujoco_warp on GitHub](https://github.com/google-deepmind/mujoco_warp)
- [NVIDIA Warp documentation](https://nvidia.github.io/warp/)
- [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/)

MuJoCo Warp is developed by Google DeepMind and NVIDIA and released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This project is an independent tool and is not affiliated with or endorsed by Google DeepMind or NVIDIA.

---

## License

MIT
