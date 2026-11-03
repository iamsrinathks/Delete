# Reservation Sizer — POC Implementation Plan

> In-cluster GitOps loop that analyses GKE node demand and raises GitHub PRs to resize Compute Engine reservations defined in Terraform.
> **POC scope:** one cluster, scale-up only, human-merged PRs.

**Status:** POC v1.0 · **Owner:** `<OWNER_TEAM>` · **Supersedes for POC:** Implementation Plan v0.2

---

## 0. Parameters

Everything environment-specific is a parameter. Nothing in this document is tied to a specific cluster, project or region.

| Parameter | Meaning | Example format |
|---|---|---|
| `<PROJECT_ID>` | GCP project hosting the cluster and its reservations | `my-project-id` |
| `<CLUSTER_NAME>` | Target GKE cluster for the POC | `my-cluster` |
| `<REGION>` | Cluster region | `<region>` |
| `<ZONES>` | Zones of the regional cluster | `<zone-a>, <zone-b>, <zone-c>` |
| `<ENV>` | Environment label | `int` / `pre` / `prd` |
| `<NAMESPACE>` | Namespace for the sizer | `platform-capacity` |
| `<KSA_NAME>` | Kubernetes service account | `reservation-sizer` |
| `<GSA_NAME>` | Google service account | `reservation-sizer-poc` |
| `<INFRA_REPO>` | GitHub repo holding Terraform | `org/infra-repo` |
| `<TFVARS_PATH>` | Reservations tfvars file for this cluster | `envs/<ENV>/<PROJECT_ID>/<REGION>/reservations.auto.tfvars.json` |
| `<GITHUB_APP_NAME>` | Dedicated GitHub App for the sizer | `reservation-sizer-poc` |
| `<GITHUB_APP_SECRET>` | Secret Manager secret holding the App private key | `reservation-sizer-github-app-key` |
| `<LOG_BUCKET>` | Dedicated log bucket for autoscaler visibility logs | `ca-visibility` |
| `<LOG_VIEW>` | Log view on that bucket | `ca-visibility-view` |
| `<LOG_LOCATION>` | Log bucket location | `global` or `<REGION>` |
| `<EGRESS_PROXY>` | Egress proxy for GitHub access (if used) | `proxy.internal:3128` |
| `<CODEOWNERS_TEAM>` | Team that approves PRs | `@org/platform-team` |
| `<SCHEDULE>` | CronJob schedule | `0 6 * * *` |

> Keep these in one place, for example a Config Sync–managed `ConfigMap/sizer-env` plus Terraform variables, so rolling out to another cluster is a parameter change, not a code change.

---

## 1. Scope

| In the POC | Parked (tweak backlog, §9) |
|---|---|
| **One cluster** (`<CLUSTER_NAME>`), system + user pools | Remaining clusters |
| **Scale-up only** via PR; **human merges every PR** | Auto-merge, automated scale-down |
| **Per-zone Terraform** (zone → machine type) with zero-replacement migration | Seasonality floor, future reservations |
| **Stockout-only** unserved-demand signal | Native GKE pending-pod / provisioning metrics |
| **Custom least-privilege IAM** + log view | Moving the sizer out of the cluster |
| **Dedicated GitHub App**, key in Secret Manager | HCP Terraform run-status integration |
| Simple feedback: desired vs actual count | Automated retries, change-record integration |

**Why these cuts:**
- **No auto-merge:** removes the auto-merge vs four-eyes conflict.
- **No automated scale-down:** releasing reserved capacity is a one-way door, so the POC never does it.
- **No HCP token:** one less secret in the cluster.

---

## 2. Architecture

```
<CLUSTER_NAME> (system node pool)
 └─ CronJob reservation-sizer  (namespace: <NAMESPACE>, schedule: <SCHEDULE>)
      1. Collect ─► 2. Compute ─► 3. Render tfvars ─► 4. Raise PR (draft in dry-run)
         │
         ├─ Managed Prometheus   ← kube-state-metrics (nodes only)
         ├─ Log view <LOG_VIEW>  ← cluster autoscaler visibility logs
         ├─ Compute API          ← reservations (count, in-use)
         ├─ GKE operations API   ← node pool upgrade windows
         └─ Kubernetes API       ← current node labels

<INFRA_REPO> (human merge) ─► HCP Terraform apply ─► Compute Engine reservations
```

---

## 3. Build steps

### Step 1 — Terraform refactor (Week 1)

- Move to `<TFVARS_PATH>`, keyed **zone → machine type**:

```json
{
  "reservations": {
    "<ZONE_A>": {
      "<MACHINE_TYPE>": {
        "name": "<EXISTING_RESERVATION_NAME_ZONE_A>",
        "count": 0,
        "pools": { "system": 0, "user": 0 }
      }
    },
    "<ZONE_B>": {
      "<MACHINE_TYPE>": {
        "name": "<EXISTING_RESERVATION_NAME_ZONE_B>",
        "count": 0,
        "pools": { "system": 0, "user": 0 }
      }
    }
  }
}
```

- `name` comes from the file, so **existing reservations keep their names**. Add `moved {}` blocks for any address changes.
- `pools` is informational only; Terraform reads `count`.
- Add `lifecycle { prevent_destroy = true }` to the reservation resource.
- **Exit check:** `terraform plan` shows **0 to add, 0 to destroy**; only in-place updates or no changes.
- **[Verify]** Is any reservation **attached to a commitment (CUD)**? If yes, keep it **out of** the sizer's file for the POC. Commitment-attached reservations follow a separate replace process.

### Step 2 — Data plumbing (Week 1)

#### 2a. kube-state-metrics (nodes only)

- Deploy via Config Sync with:
  - `--resources=nodes` (no pod list/watch, so no tenant pod data)
  - `--metric-labels-allowlist=nodes=[topology.kubernetes.io/zone,node.kubernetes.io/instance-type,cloud.google.com/gke-nodepool]`
- Scrape with a `PodMonitoring` at a 60s interval.

Peak node count per zone / type / pool ([Verify] in the Managed Prometheus query UI):

```promql
max_over_time(
  count by (label_topology_kubernetes_io_zone,
            label_node_kubernetes_io_instance_type,
            label_cloud_google_com_gke_nodepool)
    (kube_node_labels{cluster="<CLUSTER_NAME>"})[<WINDOW>:5m]
)
```

For the upgrade-window exclusion, run the inner `count by (...)` as a **range query** (5m step) and drop samples inside upgrade windows **in code**.

#### 2b. Autoscaler stockout signal (via log view)

- **Log sink** → bucket `<LOG_BUCKET>` with filter:

```
resource.type="k8s_cluster"
resource.labels.cluster_name="<CLUSTER_NAME>"
LOG_ID("container.googleapis.com/cluster-autoscaler-visibility")
```

- **Log view** `<LOG_VIEW>` on `<LOG_BUCKET>`.
- Grant `roles/logging.viewAccessor` with an **IAM condition** limited to that view:

```
resource.name == "projects/<PROJECT_ID>/locations/<LOG_LOCATION>/buckets/<LOG_BUCKET>/views/<LOG_VIEW>"
```

- The sizer reads **only**:
  - `eventResult` entries where `errorMsg.messageId = "scale.up.error.out.of.resources"`
  - the matching `scaleUp` event (join on `eventId`) to get `increasedMigs[].mig.zone`
- **Ignored:** quota errors, IP-exhaustion errors, and all `noScaleUp` reasons. These aren't fixable by reservations, and tenants can trigger them with misconfigured pods.
- **Requirement:** Cloud Logging must be enabled on the cluster, or these events aren't produced.

### Step 3 — Identity and access (Week 1–2)

**Workload Identity:** `<NAMESPACE>/<KSA_NAME>` → `<GSA_NAME>@<PROJECT_ID>.iam.gserviceaccount.com`

**Custom IAM role** (instead of broad viewer roles). [Verify] permission names in your org:

```yaml
title: Reservation Sizer Reader (POC)
stage: GA
includedPermissions:
  - compute.reservations.get
  - compute.reservations.list
  - container.clusters.get
  - container.operations.get
  - container.operations.list
  - monitoring.timeSeries.list   # PromQL via Managed Prometheus; fall back to roles/monitoring.viewer if queries return 403
```

| Access | Grant |
|---|---|
| Reservations, operations, metrics | Custom role above, on `<PROJECT_ID>` |
| Autoscaler logs | Conditional `roles/logging.viewAccessor` on `<LOG_VIEW>` only |
| GitHub App key | `roles/secretmanager.secretAccessor` on `<GITHUB_APP_SECRET>` only |
| Kubernetes API | `ClusterRole`: `get`, `list` on `nodes` only |

**GitHub App `<GITHUB_APP_NAME>`:**
- **Dedicated** — not shared with other automation.
- Permissions: `contents: write`, `pull_requests: write` only.
- Installed on `<INFRA_REPO>` only.
- Private key in Secret Manager (`<GITHUB_APP_SECRET>`), fetched at runtime. **No Kubernetes Secret.**

**Network:**
- Default-deny egress for `<NAMESPACE>`.
- Allow Google APIs (Private Google Access) and GitHub via `<EGRESS_PROXY>`.
- [Verify] Standard NetworkPolicy is IP-based. If FQDN filtering is required, use your egress proxy or GKE FQDN network policy.

**Pod hardening:**
- Runs on the system pool with a `PriorityClass`.
- Non-root, read-only root filesystem, `allowPrivilegeEscalation: false`, all capabilities dropped.
- Must pass OPA Gatekeeper in deny mode.

**CronJob settings:** `concurrencyPolicy: Forbid`, `activeDeadlineSeconds`, `backoffLimit: 1`.

### Step 4 — Sizer logic (Week 2)

#### Policy (`ConfigMap/sizer-policy`, Config Sync–managed, read-only for the sizer)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sizer-policy
  namespace: <NAMESPACE>
data:
  policy.yaml: |
    enabled: true
    mode: dry-run              # dry-run → draft PR labelled do-not-merge; live → normal PR
    cluster: <CLUSTER_NAME>
    project: <PROJECT_ID>
    region: <REGION>
    env: <ENV>
    tfvarsPath: <TFVARS_PATH>
    pools: [system, user]
    window: 7d
    buffer: 10%                # matches the current manual rule for comparison
    surgeHeadroomPerZone: 1    # MAX across pools, not sum (pools upgrade one at a time)
    stockoutAddPerZone: 1      # only if out.of.resources was seen in that zone in the window
    stockoutCapPerZone: 2
    maxStepPerZonePerPR: 3
    minDataCoverage: 0.9       # below this → no PR, alert instead
```

#### Calculation (per zone, per machine type)

```
for each pool on that machine type:
    peak[pool]     = max node count over window, excluding upgrade windows
    required[pool] = ceil(peak[pool] × (1 + buffer))

target = Σ required[pool]
       + surgeHeadroomPerZone                          # once per zone, not per pool
       + min(stockoutAddPerZone, stockoutCapPerZone)   # only if a stockout was seen in this zone

increase = clamp(target − current, 0, maxStepPerZonePerPR)
```

#### Guardrails (in code)

- **Up only.** If the target is below the current count, add an **"advisory: could reduce by N"** line to the PR body. Never change the number down.
- **Fail-safe.** Any query error, data coverage below `minDataCoverage`, or an empty series means **no PR**, `sizer_last_run_success=0`, and an alert. Missing data must never produce a number.
- **One open PR at a time.** Update the existing PR rather than opening a new one.
- **No stacking.** If the count on the default branch is ahead of the actual reservation count, skip the run and alert.
- **Kill switch.** `enabled: false` → the run exits immediately.

#### Exposure KPI (reported in every PR)

```
on_demand_nodes_now[zone, type] = node_count (Kubernetes API) − reservation_in_use (Compute API)
```

[Verify] the in-use field name on the reservation resource.

### Step 5 — PR and checks (Week 2–3)

**PR body template:**

```markdown
## Reservation Sizer — <ENV> / <PROJECT_ID> / <REGION> / <CLUSTER_NAME>

| Zone | Machine type | Current | Proposed | Δ | Driver |
|---|---|---|---|---|---|
| <ZONE> | <MACHINE_TYPE> | <N> | <N> | +<N> | peak per pool, +buffer, +surge, +stockout |

**Evidence**
- Window: <WINDOW> · Excluded upgrade operations: <OPERATION_IDS>
- Stockout events (out.of.resources) by zone: <...>
- On-demand exposure now (nodes − reservation in-use): <...>
- Advisory reductions (not applied): <...>

**Previous change status:** <applied ✅ | not applied after 24h in <ZONE> ⚠️>
**Data coverage:** <PERCENT> · **Input snapshot hash:** <SHA256>
Generated by reservation-sizer <VERSION> · run <RUN_ID>
```

**Required CI check (script)** — fails unless:
- the diff touches **only** `<TFVARS_PATH>`;
- only `count` values change;
- every change is ≤ `maxStepPerZonePerPR` and **positive**.

**Speculative plan:** must show only **in-place updates** to reservation resources (no `-/+`, no create or destroy).

**CODEOWNERS:**

```
<TFVARS_PATH>  <CODEOWNERS_TEAM>
```

A human approves and merges every PR.

### Step 6 — Feedback, lite (Week 3)

- Each run compares the **count on the default branch** with the **actual reservation count** from the Compute API.
- If they still differ **24h after merge**, raise an alert and add a line to the next PR body.
- The failure reason (for example `RESERVATION_CAPACITY_LIMIT_EXCEEDED`, `ZONE_RESOURCE_POOL_EXHAUSTED`) is read **manually** from the HCP Terraform run.

### Step 7 — Run and evaluate (Weeks 3–4)

- **1–2 weeks in `dry-run`** (draft PRs labelled `do-not-merge`).
- Then **`live`**, still human-merged.

---

## 4. Self-observability

| Metric | Purpose |
|---|---|
| `sizer_recommended_count{zone,machine_type}` | What the sizer wants |
| `sizer_reserved_count{zone,machine_type}` | What exists |
| `sizer_on_demand_exposure{zone,machine_type}` | Nodes running outside reservation |
| `sizer_stockout_events{zone}` | `out.of.resources` count in window |
| `sizer_drift{zone,machine_type}` | Default branch ≠ actual |
| `sizer_data_coverage` | Fraction of window with data |
| `sizer_last_run_success` / `sizer_last_run_timestamp` | Liveness |

---

## 5. Testing

| Test | Expected result |
|---|---|
| **Unit** — peak, upgrade exclusion, per-pool rounding then sum, surge = max not sum, clamp, up-only | Golden-file tfvars match |
| **Fail-safe** — block Prometheus or Logging access for one run | No PR, alert fired |
| **Tenant noise** — deploy an unschedulable pod (bad nodeSelector) in a test namespace | Recommendation unchanged |
| **Stockout fixture** — inject an `out.of.resources` event for one zone | Only that zone gets the add, capped |
| **CI guard** — PR that edits another file or decreases a count | Check fails |
| **Upgrade** — observe one real node pool upgrade during the window | Surge nodes excluded from peak |

---

## 6. POC success criteria

- [ ] Zero-replacement Terraform migration completed.
- [ ] Dry-run recommendations within **±1 node per zone** of the manual "7-day peak + 10%".
- [ ] At least **one real PR merged and applied**, confirmed by the feedback check.
- [ ] One node pool upgrade observed with surge nodes correctly excluded.
- [ ] Fail-safe and tenant-noise tests passed.
- [ ] Security team has reviewed the POC design (identity, log view, GitHub App scope) before `live`.

---

## 7. Accepted POC risks

| Risk | POC mitigation | Proper fix (backlog) |
|---|---|---|
| GitHub App credential fetched in-cluster | Dedicated App, Secret Manager, path-restricted CI check | Move the sizer out of the cluster |
| ANY affinity — system pool not protected from user-pool bursts | Monitor exposure KPI | SPECIFIC reservation for system pool |
| No zone-loss (N+1) or region failover sizing | Documented | Costed N+1 decision |
| 7-day window misses month/quarter/year-end peaks | Acceptable in non-prd | Seasonal floor + business-event calendar |
| No automated retries of failed applies | Manual via drift alert | HCP integration with defined owner |

---

## 8. Open items before Step 1

- [ ] Confirm `<CLUSTER_NAME>`, `<PROJECT_ID>`, `<REGION>`, `<ZONES>` for the POC.
- [ ] Any reservation attached to a **CUD**?
- [ ] Does GitHub egress go through a proxy (`<EGRESS_PROXY>`)?
- [ ] Is Cloud Logging enabled on the cluster?
- [ ] Which HCP Terraform workspace owns the reservations?
- [ ] Are there other node pools on the same machine types (they would also consume ANY reservations)?
- [ ] Is Config Sync drift prevention enabled?

---

## 9. Tweak backlog (post-POC, rough priority)

1. **Move the sizer out of the cluster** — one scheduled job in a platform tooling project serving all clusters.
2. **SPECIFIC reservation for the system pool.**
3. **Seasonal floor and business-event calendar** for prd.
4. **N+1 zonal sizing** decision (costed or formally risk-accepted).
5. **Native GKE pending-pod and provisioning-failure metrics** (GKE 1.37+).
6. **HCP Terraform integration** for apply status and retries.
7. **Break-glass kill switch**, Binary Authorization, change-record linkage, cost cap per PR.
8. **Auto-merge in lower envs** via a separate approver identity.
9. **Roll out** to remaining clusters by changing parameters only.

---

## 10. References

- View cluster autoscaler events and metrics — https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-autoscaler-visibility
- About GKE cluster autoscaling — https://docs.cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler
- Modify reservations — https://docs.cloud.google.com/compute/docs/instances/reservations-modify
- Monitor consumption of reservations — https://docs.cloud.google.com/compute/docs/instances/reservations-monitor
- Troubleshooting reservation updates — https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-reservation-updates
- Configure log views on a log bucket — https://docs.cloud.google.com/logging/docs/logs-views
- Cloud Logging roles and permissions — https://docs.cloud.google.com/iam/docs/roles-permissions/logging
- GKE role-based access control — https://cloud.google.com/kubernetes-engine/docs/role-based-access-control
- Kube State Metrics (Managed Prometheus exporter) — https://docs.cloud.google.com/stackdriver/docs/managed-prometheus/exporters/kube_state_metrics
- Prevent config drift (Config Sync) — https://docs.cloud.google.com/anthos-config-management/docs/how-to/prevent-config-drift
