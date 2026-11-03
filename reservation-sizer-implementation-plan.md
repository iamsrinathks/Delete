# Reservation Sizer — Implementation Plan

> In-cluster GitOps loop that analyses GKE node demand trends and raises GitHub PRs to resize Compute Engine reservations defined in Terraform.

**Status:** Draft v0.2 · **Owner:** Platform Team · **Date:** 2026-11-03

---

## 0. Changelog

### v0.2 (2026-11-03) — self-review fixes

| # | Change | Why |
|---|---|---|
| 1 | **Demand = observed nodes + unserved demand** | Observed peaks are censored: nodes that couldn't be created (stockout) never appear, and reservation **Used ≤ Reserved** by definition. Sizing on observed data alone sizes to "what we got", not "what we needed". |
| 2 | **Node-count history from kube-state-metrics** (self-deployed, via Managed Prometheus) | v0.1 named no concrete node-count metric. GKE's managed KSM package is a curated workloads/storage set; the self-deployed integration keeps `kube_node_*`. |
| 3 | **Autoscaler signals from Cloud Logging**, not Kubernetes events | Visibility events (`scaleUp`, `noScaleUp`) are Cloud Logging entries, separate from Kubernetes events. Added `roles/logging.viewer`. |
| 4 | **Feedback checker moved to Phase 1 (core)** | Raising an auto-consumed reservation to cover running on-demand nodes is the **normal** case — and is exactly what gets rejected when the zone lacks reservation capacity. |
| 5 | **Bot state in an unmanaged ConfigMap** (or stateless) | Config Sync self-healing reverts drift on managed objects; drift prevention (if enabled) rejects the write. |
| 6 | **Upgrade windows from GKE operations API** | v0.1's "surge marker" was not a documented mechanism. Added `roles/container.viewer`. |
| 7 | **Terraform migration keeps existing reservation names** | `moved {}` only avoids replacement if `name` is unchanged; a name change replaces live reservations. |

### v0.1 (2026-11-02) — initial draft

---

## 1. Scope and decisions so far

| Area | Decision |
|---|---|
| **Deployment model** | **One instance per cluster** (svp01/svp02 × europe-west2/europe-west1) |
| **Approval** | **Human approval mandatory in prd**; auto-merge allowed in lower envs (int/pre) |
| **Terraform layout** | **Option A — per-zone counts**, keyed by `zone → machine type` |
| **Pools in scope** | **System** and **user** node pools only |
| **Sizing unit** | **(zone, machine type)** — pools share machine types in some envs |
| **Non-GKE VMs** | None in the projects; drift check still added as a guard |
| **Reservation consumption** | Automatically consumed (node pools use **ANY** affinity) |

**Out of scope (v1):** future reservations, ComputeClasses, GPU/accelerator pools, cross-region failover sizing.

---

## 2. Architecture overview

```
 ┌────────────────────────── GKE cluster (per cluster) ──────────────────────────┐
 │  system node pool                                                             │
 │  ┌──────────────────────────────────────────────────────────────────────────┐ │
 │  │ CronJob: reservation-sizer  (namespace: platform-capacity)              │ │
 │  │                                                                          │ │
 │  │  0. Feedback ─► 1. Collector ─► 2. Policy ─► 3. Renderer ─► 4. Guards ─┐ │ │
 │  │     checker          ▲             engine                               │ │ │
 │  │                      │      ConfigMap: sizer-policy (Config Sync)       │ │ │
 │  │                      │      ConfigMap: sizer-state  (UNMANAGED)         │ │ │
 │  └──────────────────────┼──────────────────────────────────────────────────┼─┘ │
 │   kube-state-metrics ───┘ (self-deployed, scraped by Managed Prometheus)  │   │
 └────────────────────────────────────────────────────────────────────────────┼───┘
   Inputs: Managed Prometheus · Cloud Logging · Compute API ·                 ▼
           GKE operations API · Kubernetes API                    5. PR raiser (existing)
                                                                             │
                                         GitHub repo ◄────────────────────────┘
                                              │ merge (manual in prd, auto in lower envs)
                                              ▼
                                       HCP Terraform run ──► Compute Engine reservations
```

**Why a CronJob, not a controller:** sizing is a **slow loop** (days, not seconds). History comes from Managed Prometheus, so the job doesn't need to sample continuously.

---

## 3. Components

### 3.0 Feedback checker (runs first, every run)

- Read the **HCP Terraform run status** for the last merged sizer PR.
- Compare **desired vs actual** reserved count per (zone, machine type) via the Compute API.
- Classify failures:
  - `RESERVATION_CAPACITY_LIMIT_EXCEEDED` — zone lacks reservation capacity for the family (expected to be common when raising a reservation over running on-demand nodes).
  - `ZONE_RESOURCE_POOL_EXHAUSTED` — zonal stockout.
- Behaviour:
  - **Retry with backoff** on later runs (capacity changes over time).
  - After N failed attempts → **"zone stuck" alert** to on-call; include in the next PR body.
  - **Don't stack** new increases on top of an unapplied one — recompute from actual state.
- Drift checks:
  - **desired ≠ actual** persisting beyond threshold → alert.
  - **Used > GKE node count** for a (zone, type) → unexpected VM consuming the reservation → alert.

### 3.1 Collector

| Signal | Source | Purpose |
|---|---|---|
| **Node count history** per pool/zone/type | **kube-state-metrics** (`kube_node_labels`, `kube_node_info`) via Managed Prometheus (PromQL) | Observed demand |
| **Unschedulable pods** | kube-state-metrics pod status (pending / unschedulable) | Unserved demand |
| **Autoscaler visibility events** (`scaleUp`, `noScaleUp`, `eventResult`) | **Cloud Logging**, log `container.googleapis.com/cluster-autoscaler-visibility` | Unserved demand + failure reasons |
| **Reservation Reserved / Used** | Cloud Monitoring reservation metrics | Current headroom only — **never as demand** (Used ≤ Reserved) |
| **Reservations inventory** | Compute API | Name, machine type, count, consumption type |
| **Node pool upgrade windows** | **GKE operations API** (node pool upgrade operations, start/end time) | Exclude surge nodes from peaks |
| **Current node labels** | Kubernetes API (`topology.kubernetes.io/zone`, `node.kubernetes.io/instance-type`, `cloud.google.com/gke-nodepool`) | Map pool → zone → machine type |

**Notes and caveats:**
- **kube-state-metrics:** GKE's managed KSM package is a curated set (workloads state / storage) and may not include node metrics → use the **self-deployed KSM integration** for `kube_node_*`. Ingestion is **charged per sample** — scrape node metrics only, at a modest interval.
- **Autoscaler events:** only produced when **Cloud Logging is enabled** on the cluster. `noScaleUp` is **best-effort and throttled** → use as a **flag** ("demand went unserved in zone X"), not a precise count.
- **Reservation metrics:** project-scoped (fine for one-instance-per-cluster); refreshed at least every 30 minutes.

### 3.2 Policy engine

Per **pool** per **zone**:

```
observed_peak  = max(node_count over window, excluding upgrade windows)
unserved       = estimate of nodes needed for unschedulable pods / noScaleUp periods
                 (v1: +1 node per zone per day with unserved demand, capped — refine in dry run)
demand_peak    = observed_peak + unserved
required       = ceil(demand_peak × (1 + buffer)) + surgeHeadroom
required       = clamp(required, floor, current ± maxStepPerPR)
```

Then **aggregate per (zone, machine type)**:

```
reservation[zone][type] = Σ required(pool)   for pools on that type
```

- Round **per pool before summing** so the small system pool isn't lost in rounding.
- Pools differing on other consumption-matching properties are **separate keys** (full property list to be confirmed — see §12).
- **Hysteresis:** preferred approach is **stateless** — recompute the recommendation over the last N run windows from Prometheus history. Fallback: state in **`ConfigMap/sizer-state`** (not managed by Config Sync).

### 3.3 Renderer

- Reads/writes **only** this cluster's file:
  `envs/<env>/<project>/<region>/reservations.auto.tfvars.json`
- Never regex-edits HCL.
- **Deterministic, sorted JSON** for clean diffs.
- Preserves the **existing reservation name** field for each key (see §5).

### 3.4 Guardrails

- **Max step per PR** (absolute and %).
- **Cooldown** after any merged change.
- **Change-freeze calendar** → no PRs, or PRs labelled `do-not-merge`.
- **One open PR per file** — update the existing PR.
- **No stacking** while a previous change is unapplied (feedback checker).
- **Dry-run mode**, **kill switch** (`enabled: false`).
- **Never auto-delete** a reservation key; flag in PR body only.

### 3.5 PR raiser (existing)

- Labels: `reservation-sizer`, `env:<env>`, `direction:up|down`, `auto-merge` (lower envs, within limits only).
- PR body template in §6.

### 3.6 Self-observability

- `sizer_recommended_count{zone,machine_type}`
- `sizer_reserved_count{zone,machine_type}`
- `sizer_headroom{zone,machine_type}`
- `sizer_unserved_demand{zone,pool}`
- `sizer_apply_failures_total{zone,reason}`
- `sizer_drift{zone,machine_type}`
- `sizer_last_run_success` / `sizer_last_run_timestamp`

---

## 4. Policy configuration

`ConfigMap/sizer-policy` — **managed by Config Sync** (read-only for the bot):

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sizer-policy
  namespace: platform-capacity
data:
  policy.yaml: |
    enabled: true
    dryRun: true
    env: pre                      # int | pre | prd
    autoMerge: true               # false in prd
    freezeCalendar: freeze-windows.yaml
    feedback:
      maxApplyRetries: 3
      driftAlertAfter: 24h
    unservedDemand:
      nodesPerZonePerEventDay: 1  # v1 heuristic — tune in dry run
      maxExtraPerZone: 3
    pools:
      system:
        window: 30d
        buffer: fixed-1-per-zone  # +1 node per zone, not a %
        surgeHeadroomPerZone: 1
        scaleDown: never          # manual PR only
        maxStepPerPR: 2
      user:
        window: 7d
        prdWindow: 35d            # catch month-end peaks — confirm retention (§12)
        buffer: 10%
        surgeHeadroomPerZone: 1   # matches maxSurge=1; NOT valid for blue-green
        scaleDown:
          consecutiveRuns: 3
          maxStepPercent: 10
          cooldown: 14d
        maxStepPerPR: 5
```

`ConfigMap/sizer-state` — **not** in the Config Sync repo (only if the stateless approach isn't used). If Config Sync manages the namespace, ensure this object is not declared in the source of truth.

> Policy values are **starting recommendations**, not Google guidance. Tune during dry run.

---

## 5. Terraform changes (Option A)

**File schema** — `reservations.auto.tfvars.json`:

```json
{
  "reservations": {
    "europe-west2-a": {
      "n2-standard-8": {
        "name": "<existing-reservation-name-zone-a>",
        "count": 14,
        "pools": { "system": 3, "user": 11 }
      }
    },
    "europe-west2-b": {
      "n2-standard-8": {
        "name": "<existing-reservation-name-zone-b>",
        "count": 13,
        "pools": { "system": 3, "user": 10 }
      }
    }
  }
}
```

**Module changes:**

- [ ] `variable "reservations"` — map of zone → map of machine type → object(`name`, `count`, `pools`).
- [ ] Flatten to `zone × type`; `for_each` with stable keys `"${zone}/${type}"`.
- [ ] **`name` comes from the variable**, so existing reservations **keep their current names**. New keys get a naming convention.
- [ ] `pools` is informational only — Terraform reads `count`.
- [ ] **Migration:** `moved {}` blocks from current addresses to new keys. **Acceptance criterion: `terraform plan` shows zero replacements** (any `-/+` on a reservation = stop).
- [ ] Confirm which **HCP Terraform workspace** owns reservations (prj vs gke).

---

## 6. PR template

```markdown
## Reservation Sizer — <env> / <project> / <region>

| Zone | Machine type | Current | Proposed | Δ | Driver |
|---|---|---|---|---|---|
| europe-west2-b | n2-standard-8 | 11 | 13 | +2 | user pool observed peak 11 + unserved 1 → 12 (+10%, +1 surge) |

**Evidence**
- Window: 7d (excluded upgrade windows: <operation ids / dates>)
- Observed peak per pool/zone: ...
- Unserved demand: <noScaleUp days / unschedulable pods> per zone
- Reserved vs Used (last 24h): ...

**Previous change status:** <applied ✅ | failed: RESERVATION_CAPACITY_LIMIT_EXCEEDED in zone b (attempt 2/3)>
**Guardrails applied:** maxStep ✅ · cooldown ✅ · freeze ✅ · no-stacking ✅
**Flags:** <e.g. unused reservation for retired machine type — manual review>

Generated by reservation-sizer <version> · run <id>
```

---

## 7. Security and access

**Workload Identity** — KSA `platform-capacity/reservation-sizer` → GSA `reservation-sizer@<project>`.

| Permission | Why |
|---|---|
| `roles/monitoring.viewer` | Managed Prometheus (PromQL) + reservation metrics |
| `roles/logging.viewer` | Cluster autoscaler visibility events |
| `roles/compute.viewer` | Reservations inventory and counts |
| `roles/container.viewer` | GKE operations (upgrade windows) |
| HCP Terraform **read-only** team token | Run status for feedback (stored in Secret Manager) |
| GitHub App (existing) | Open/update PRs — **scoped to the infra repo only** |

**Kubernetes RBAC:**
- `ClusterRole`: `get/list` on `nodes`.
- `Role` in `platform-capacity`: `get/update` on `configmaps/sizer-state` only (if stateful fallback used).

**Never** grant permission to modify reservations — **Terraform is the only writer**.

**Pod hardening:** system pool + `PriorityClass`; non-root, read-only root FS, no privilege escalation; `NetworkPolicy` egress only to Google APIs, GitHub, HCP Terraform; must pass **OPA Gatekeeper** (deny mode).

**kube-state-metrics:** deployed via Config Sync; scrape config limited to node (and pod-status) metrics to control ingestion cost.

---

## 8. GitHub controls

- **CODEOWNERS** per env path, e.g. `envs/prd/** @platform-leads`.
- **Branch protection** on prd paths: required review + required `terraform plan` check.
- `auto-merge` label applied **only** by the app, **only** in lower envs within limits.
- Required check: speculative plan shows **only reservation count changes** and **no replacements**.

---

## 9. Delivery phases

### Phase 0 — Foundations (1–2 weeks)
- [ ] Deploy **kube-state-metrics** (node metrics) + confirm PromQL queries in tr1-pre.
- [ ] Confirm **Cloud Logging** enabled and autoscaler visibility events queryable.
- [ ] Confirm reservation metric type names and **retention** for a 35-day window.
- [ ] Terraform refactor to Option A with `moved {}` — **zero-replacement plan**.
- [ ] Workload Identity, GSA + 4 viewer roles, RBAC, NetworkPolicy via Config Sync.

### Phase 1 — Dry run + feedback (2–4 weeks)
- [ ] Collector, policy engine (incl. **unserved demand**), renderer — `dryRun: true`.
- [ ] **Feedback checker live** (reads HCP runs + actual counts; alerts only).
- [ ] Compare recommendations vs current manual **7-day peak + 10%**.
- [ ] Validate upgrade-window exclusion against a real node pool upgrade.
- [ ] Tune the unserved-demand heuristic against any real stockout periods.
- [ ] Dashboard for `sizer_*` metrics.

### Phase 2 — Lower envs live (int → pre)
- [ ] PR creation + `auto-merge` in int, then pre.
- [ ] Exercise retry/backoff and "zone stuck" alerting.
- [ ] Run through at least **one node pool upgrade** and **one month-end**.

### Phase 3 — prd (manual approval)
- [ ] Enable with `autoMerge: false`.
- [ ] Raise through **change/CAB**; document rollback.
- [ ] Runbook: kill switch, reverting a merged PR, handling a stuck zone.

### Phase 4 — Hardening
- [ ] **Transition mode** for machine type migrations (old + new reservations during overlap).
- [ ] Blue-green upgrade support (surge headroom ≈ pool size, if adopted).
- [ ] Optional: Compute Engine **idle/underutilised reservation recommendations** as an extra scale-down signal.
- [ ] Optional: tenant **ResourceQuota** totals as a leading growth indicator.

---

## 10. Testing

| Level | What |
|---|---|
| **Unit** | Peak calc, upgrade exclusion, unserved-demand add-on, per-pool rounding then sum, hysteresis, guardrails, no-stacking |
| **Golden files** | Fixed input series → expected tfvars JSON |
| **Integration** | tr1-pre against real APIs in dry run |
| **Failure drills** | Prometheus/Logging/Compute API down, GitHub down, HCP apply failure, `RESERVATION_CAPACITY_LIMIT_EXCEEDED` |
| **Censoring test** | Simulated stockout (nodes capped, pods pending) → recommendation must **exceed** observed peak |
| **Chaos** | Single-zone stockout skewing nodes → per-zone recommendation correct |

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Censored demand** (stockouts hide real need; Used ≤ Reserved) | Unserved-demand signal; never use Used as demand |
| **Merged PR ≠ capacity** (increase rejected / stockout) | Feedback checker in Phase 1, retry/backoff, "zone stuck" alert |
| **Ratchet from surge nodes** | Exclude upgrade windows from GKE operations |
| **Uneven zone spread** (BALANCED is best-effort) | Per-zone sizing (Option A) |
| **Month-end peaks missed** | 35-day prd window (retention to confirm) |
| **Shared machine type miscount** | Aggregate per (zone, type) after per-pool policy |
| **Live reservations replaced during refactor** | Keep names; zero-replacement plan as acceptance criterion |
| **Bot state reverted by Config Sync** | Stateless hysteresis, or unmanaged `sizer-state` |
| **KSM ingestion cost** | Scrape node/pod-status metrics only |
| **Cost of idle buffer** | Reserved vs Used dashboard; monthly FinOps review |
| **App bug proposes bad numbers** | maxStep, CODEOWNERS, branch protection, dry run, kill switch |
| **Loop runs in cluster it sizes** | System pool + PriorityClass; not on the critical path |
| **Reservation ≠ pod scheduling** | Out of scope — taints, affinity, PDBs, requests still apply |

---

## 12. Open items

- [ ] Exact reservation **metric type names** in Cloud Monitoring.
- [ ] **Retention** of the queried series for a 35-day window.
- [ ] Full list of VM properties that must match for consumption (*About reservations*).
- [ ] Zone count for the **europe-west1** clusters.
- [ ] Do machine types change during upgrades? Is **blue-green** used anywhere?
- [ ] Which HCP Terraform workspace owns the reservations (prj vs gke)?
- [ ] Is Config Sync **drift prevention** enabled on the cluster? (Affects where bot state can live.)

---

## 13. References

- Monitor consumption of reservations — https://docs.cloud.google.com/compute/docs/instances/reservations-monitor
- Troubleshooting reservation updates — https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-reservation-updates
- Troubleshooting reservation creation — https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-reservation-creation
- Reservation recommendations overview — https://docs.cloud.google.com/compute/docs/instances/reservation-recommendations-overview
- About GKE cluster autoscaling — https://docs.cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler
- View cluster autoscaler events — https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-autoscaler-visibility
- Kube State Metrics (Managed Prometheus exporter) — https://docs.cloud.google.com/stackdriver/docs/managed-prometheus/exporters/kube_state_metrics
- Configure kube state metrics package (GKE) — https://docs.cloud.google.com/kubernetes-engine/docs/how-to/kube-state-metrics
- Prevent config drift (Config Sync) — https://docs.cloud.google.com/anthos-config-management/docs/how-to/prevent-config-drift
- Project source index (reservations, upgrade surge, ComputeClasses) — https://claude.ai/artifact/FR9pXYuBe99kUo3SBgZGVV
