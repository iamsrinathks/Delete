# Reservation Sizer — POC Implementation Steps (v1.2)

> Daily job that reads GKE node history, autoscaler stockouts and live reservations, then raises **one
> GitHub PR** that raises `instance_count` in your `node_pool_reservations` Terraform map.
> **POC:** one cluster · scale-up only · a human merges every PR.

**Status:** POC v1.2 · supersedes POC plan v1.1 · kit version `poc-1.2.0`

---

## What changed from v1.1

| Area | v1.1 | v1.2 (this kit) | Why |
|---|---|---|---|
| Stockout lookup | Crashed (`KeyError`, exit 4) when an error did not name its MIG and the scale-up touched several zones | Fixed. Such events are **reported, not added** (`stockoutAmbiguous: "skip"`; `"all-zones"` counts every zone) | BALANCED scale-ups often touch all three zones at once |
| Holds | Any held key stopped the whole run | Apply pending/failed, reservation missing or not READY → **only that key** is held; the other keys are still sized (exit 3). Live **above** the branch (DRIFT) still stops everything | A stock-out in one zone no longer freezes the other zones. Drift stays global: any apply would shrink the drifted reservation |
| Zero nodes | `node-metric`: a pool at 0 nodes looked like missing data → fail-safe | Coverage is judged **for the cluster** (any node series); a MIG without nodes counts as 0 | Pools with min 0, or a quiet zone, can be sized |
| `--ca-bundle` | Replaced the public roots | **Added to** the public roots (sizer and GitHub client) | Google APIs go direct, GitHub through a TLS-inspecting proxy, in the same run |
| CI guard | Ran the PR's own copy of the checker; diffed against `pull_request.base.sha` | Runs the checker **from the base commit**; diffs against the merge commit's first parent | A PR (or a stolen bot token) could rewrite the checker; a stale base blamed others' merges on the PR |
| Dry-run PRs | Draft + label, nothing enforced | `[DRY RUN]` title + draft + `do-not-merge`; the guard **fails** on the title or the label and re-runs on label changes | "Ready for review" + merge used to work |
| Mode switch | The draft PR stayed open with `do-not-merge` | The open PR is **closed and replaced** when the mode changes | A dry-run PR never becomes the live one |
| Stale PRs | Left open when no change was needed; force-pushed on every run | **Closed with a comment** when no increase is needed (or on drift); **no push** when the file is unchanged | Old numbers can't be merged; approvals keep their meaning |
| Affinity / Spot | Pool affinity read but unused | NO_RESERVATION and Spot pools not counted; SPECIFIC pools count only for their named reservation; `specificReservationRequired` reservations are not grown (`NOT_CONSUMABLE`) | Never grow what the pools can't use |
| Comparison aids | — | Rows add `combined_peak` / `combined_required` (peak of the summed pools) and `target_excl_stockout` | Step 2 and Step 3b gates compare like with like |
| Out-of-scope pools | Preflight warning only | Every row lists other pools on the same machine type in that zone (`out_of_scope_pools`, auto-provisioned flagged) | Auto-provisioned and compute-class pools consume ANY reservations too |
| Job | Exit 2–5 retried once; `report.json` lost with the pod | `podFailurePolicy` fails the Job at once on 2–5; one structured **`sizer result`** log line per run | No duplicate runs; numbers survive in Cloud Logging |
| Plan check | Counted "known after apply" attributes as changes | Ignores them (listed separately) | Real plans no longer fail on computed fields |

## What changed in v1.1 (from v1.0)

| Area | v1.0 | v1.1 | Why |
|---|---|---|---|
| Node history | kube-state-metrics + Managed Prometheus | **Cloud Monitoring GKE system metric** `kubernetes.io/node/cpu/allocatable_cores`, counted per node, mapped to MIGs | No Prometheus in the cluster (Dynatrace only). Same metric your `gke_node_peaks.py` already uses. Nothing new to deploy. |
| Kubernetes API access | ClusterRole on nodes | **None** (`automountServiceAccountToken: false`, no RBAC) | Everything comes from Google APIs. |
| IAM | broad viewer roles → custom role | **4 permissions** + one conditional log-view grant | Least privilege. |
| Terraform | new zone → type schema + `moved {}` | **Your existing `node_pool_reservations` map, moved as-is** to `*.auto.tfvars.json` | Same keys, so same addresses: zero replacement risk and no `moved {}` blocks. |
| Drift | hold only if branch > live | **Hold on any mismatch**, missing reservation, or status ≠ READY | Prevents writing a value below a live reservation, which would shrink it on apply. |
| Stockout signal | log view | Same, now **scoped to cluster + region**; counts only `scale.up.error.out.of.resources` | Cloud Logging is confirmed enabled. Tenants can't inflate the number. |
| GitHub token | App token | **Down-scoped per run**: one repo, `contents` + `pull_requests` write only | Limits the damage from a stolen token. |
| CI guard | path-filtered workflow | **Always-on job** that skips non-reservation PRs | A required check with a path filter blocks every unrelated PR. |

---

## Kit layout

```
config/       sizer.env.example (ALL environment values) · policy.example.json (sizing rules)
setup/        00 preflight → 01 tfvars→JSON → 02 local run → 03 access → 04 image → 05 manifests → 06 run once
sizer/        reservation_sizer.py (stdlib only) · github_pr.py (needs PyJWT[crypto]) · Dockerfile · tests/
ci/           check_reservation_change.py · check_tf_plan.py · reservation-sizer-guard.yml
terraform/    hcl_map_to_json.py · sizer_access.tf (Terraform alternative to step 3) · module notes
k8s/templates ServiceAccount · CronJob · NetworkPolicy (ConfigMap generated from policy.json)
examples/demo offline dataset (generic names) for a demo with no GCP access
tests/        CI guards (the workflow runs in a real git repo), HCL converter, GitHub client, preflight
```

Nothing in the kit holds a project, cluster or region name. Every value comes from `config/sizer.env` and `config/policy.json`.

---

## Before you start (5 minutes, no GCP access needed)

```bash
cd reservation-sizer-poc
python3 --version                                   # 3.9+
python3 -m unittest discover -s sizer/tests         # 47 tests: sizing, fail-safe, holds, drift, affinity
python3 -m unittest discover -s tests               # 39 tests: CI guard workflow, plan check, HCL converter,
                                                    # GitHub flow, preflight, rendered manifests
                                                    # (GitHub tests need: pip install "PyJWT[crypto]";
                                                    #  guard tests need git + bash; PyYAML optional)

# Offline demo - good for the team walkthrough
python3 sizer/reservation_sizer.py --fixtures examples/demo --policy examples/demo/policy.json \
  --tfvars-file examples/demo/tfvars.json --project demo --location test-region1 --cluster c1 \
  --env-name demo --now 2026-11-04T06:00:00Z --out-dir /tmp/sizer-demo
cat /tmp/sizer-demo/pr_body.md     # shows upgrade surge excluded + stockout add in zone b
```

Then configure:

```bash
cp config/sizer.env.example config/sizer.env      # fill in: project, cluster, region, env, system pool ...
cp config/policy.example.json config/policy.json  # set "pools" to your system + user node pool NAMES
```

---

## Step 0 — Preflight (read-only, your own login)

```bash
gcloud auth login                     # if not already; the kit uses `gcloud auth print-access-token`
./setup/00-preflight.sh
```

It checks, with no writes:

- **Cluster:** logging, monitoring, Workload Identity, datapath.
- **In-scope pools:** their MIGs, and any **out-of-scope pools on the same machine type** (they also consume ANY reservations).
- **Reservations:** count, in-use, **commitment-attached**, **specificReservationRequired**.
- **Stockouts:** visibility events and stockouts over 30 days.
- **GKE operations history:** how far back it goes, for the upgrade exclusion.
- **History coverage:** for **both** sources (`node-metric` and `mig`).

**Exit gate:** `0 blocker(s)`. Act on every WARN, or accept it explicitly.

> If any reservation is **attached to a commitment**, keep it out of the JSON file for the POC. The sizer also skips it automatically (status `COMMITMENT_ATTACHED`).

---

## Step 1 — Move the map to JSON (one PR, expected plan: **No changes**)

```bash
./setup/01-tfvars-to-json.sh <path/to/workspace>/terraform.tfvars
mv <path/to/workspace>/terraform.tfvars.new <path/to/workspace>/terraform.tfvars
```

- The script writes `node_pool_reservations.auto.tfvars.json` next to `terraform.tfvars`. Keys and values are identical.
- It **refuses** expressions, lists or heredocs, so nothing is silently mis-converted.
- Add `lifecycle { prevent_destroy = true }` to the reservation resource in the module (see `terraform/reservations_module_notes.md`).
- **Check:** `node_pool_reservations` must **not** also be set as an HCP Terraform workspace variable.
- Open **one PR** with both files. The speculative plan must say **`No changes.`** Anything else means stop.
- Optional plan check, where you can produce plan JSON (CLI-driven workspace or a Run Task):

  ```bash
  terraform plan -out=plan.out && terraform show -json plan.out > plan.json
  python3 ci/check_tf_plan.py plan.json     # fails on create/delete/replace or non-count changes
  ```

- Then set `LOCAL_TFVARS_FILE` (your checkout path) and `TFVARS_PATH` (repo-relative path) in `config/sizer.env`.

> **One file, two clusters:** if the map holds both regions' keys, each cluster's sizer only touches keys in its own zones. Other keys show as `NO_POOLS_IN_SCOPE` and are left unchanged.

---

## Step 2 — First real run from your laptop (report mode: no PR, no writes)

```bash
./setup/02-run-local.sh
```

Outputs land in `sizer-out/`:

- `report.json`: every number, plus the excluded upgrade operations, stockout days, exposure and notes.
- `pr_body.md`: exactly what the PR will say.
- `proposed.tfvars.json`: shown as a diff.

**Exit gate:** for every key in `report.json` → `rows[]`:

- `combined_required + surge` is within **±1 node** of the comparable `gke_node_peaks.py` number (below);
- `target_excl_stockout` is within ±1 of your manual *7-day peak + 10%*.

The sizer's formula: per-pool `ceil(peak × 1.10)`, summed per (zone, machine type), + `max(maxSurge)` once per zone, + a capped stockout add.

Comparable `gke_node_peaks.py` run (same 7 days, upgrades excluded, 10% as growth, surge added separately):

```bash
python3 gke_node_peaks.py --project "$PROJECT_ID" --target "$CLUSTER_NAME@$CLUSTER_LOCATION" \
  --days 7 --basis peak_excl_upgrades --growth-pct 10 --skip-logs
```

- Take its `sizing_input` per shape/zone (no `--include-surge`), then add the largest `maxSurge` of the in-scope pools.
- It counts **every** pool of that shape. If a row lists `out_of_scope_pools`, its number is higher by their nodes.
- `target` can be above `combined_required + surge`. The sizer sums **per-pool** peaks and rounds each pool up, which is at least the peak of the summed pools. The PR body lists every key where the two differ. That gap is the POC's deliberate headroom, not an error.
- If the numbers differ by more, look in `rows[].pools[]` (per pool: peak, coverage, buckets with nodes, buckets excluded).

---

## Step 3 — Identity + log access (gcloud **or** Terraform, not both)

```bash
DRY_RUN=1 ./setup/03-access-gcloud.sh     # prints shell-quoted commands - paste into the change record
./setup/03-access-gcloud.sh               # idempotent
CREATE_LOG_METRIC=1 ./setup/03-access-gcloud.sh   # optional: log-based metric reservation_sizer_errors
```

If your org manages IAM in Terraform, use `terraform/sizer_access.tf` in the project/IAM workspace instead.

It creates:

| What | Scope |
|---|---|
| GSA `GSA_NAME` | — |
| Custom role: `compute.reservations.list`, `container.clusters.get`, `container.operations.list`, `monitoring.timeSeries.list` | project |
| `roles/iam.workloadIdentityUser` for `NAMESPACE/KSA_NAME` | that GSA only |
| Log bucket + sink (autoscaler visibility, **this cluster + region only**) + log view | project |
| `roles/logging.viewAccessor` with an **IAM condition on that one view** | the view only |
| `roles/secretmanager.secretAccessor` | the GitHub key secret only (when set) |
| `roles/iam.serviceAccountTokenCreator` for `LOCAL_IMPERSONATOR_MEMBER` | optional, POC only; remove afterwards |

> **The sink starts empty.** It only holds events from creation onwards, so the in-cluster stockout signal builds up over the first window. Your laptop run in Step 2 reads project logs directly, so it already sees history.

---

## Step 3b — Prove least privilege and the fail-safe (same laptop)

```bash
LOCAL_IMPERSONATE=1 ./setup/02-run-local.sh      # runs AS the GSA, reads logs through the view
SIZER_ACCESS_TOKEN=invalid ./setup/02-run-local.sh; echo "exit=$?"   # expect exit=2, no proposed file
```

- **Gate 1:** the impersonated run gives the **same `target_excl_stockout`** for every key as Step 2. That proves the 4 permissions plus the view are enough.
  - `stockout_add` can be lower: the new log bucket only holds events from when the sink was created.
  - Compare `stockout_add` again once the bucket holds `windowDays` of events.
- **Gate 2:** a bad credential produces **no recommendation**. Missing data never produces a number.

---

## Step 4 — Build and push the image

```bash
BASE_IMAGE=<approved-registry>/python:3.12-slim ./setup/04-build-push.sh   # builds linux/amd64 on Apple Silicon too
```

- Pin by digest in `IMAGE` if your policy requires it.
- The only runtime dependency is `PyJWT[crypto]`. Use `PIP_INDEX_URL` for your internal mirror.

---

## Step 4b — GitHub App, secret, CI guard (needed from Step 6: the job reads the tfvars file from GitHub)

1. **Create a dedicated GitHub App** (not the shared PR raiser):
   - Permissions: **Contents: read & write**, **Pull requests: read & write**, Metadata: read.
   - No webhook.
   - Install it on the **infra repo only**.
2. **Store the private key in Secret Manager, then delete the local copy:**

   ```bash
   gcloud secrets create "$GITHUB_APP_SECRET" --project "$PROJECT_ID" \
     --replication-policy=user-managed --locations="$CLUSTER_LOCATION" --data-file=app-key.pem
   rm -f app-key.pem
   ./setup/03-access-gcloud.sh          # re-run: adds secretAccessor on this secret only
   ```

3. **Set the IDs in `config/sizer.env`:**
   - `GITHUB_APP_ID` comes from the App page.
   - `GITHUB_APP_INSTALLATION_ID` is the number at the end of the installation settings URL (`…/settings/installations/<ID>`).
4. **Install the CI guard in the infra repo:**

   ```bash
   cp ci/check_reservation_change.py <infra-repo>/ci/
   cp ci/reservation-sizer-guard.yml  <infra-repo>/.github/workflows/
   gh variable set SIZER_TFVARS_PATH --repo "$GITHUB_REPO" --body "$TFVARS_PATH"
   gh variable set SIZER_MAX_STEP    --repo "$GITHUB_REPO" --body 3      # = policy maxStepPerKeyPerPR
   echo "/$TFVARS_PATH @<org>/<platform-team>" >> <infra-repo>/.github/CODEOWNERS
   ```

   - On the default branch, require the check **`reservation-sizer-guard`**, 1 approval and CODEOWNERS review.
   - Also turn on **"Dismiss stale pull request approvals when new commits are pushed"**.
   - Bot PRs (branch `reservation-sizer/*`) fail unless **only that file** changed, **only `instance_count`** moved, **up only**, and **≤ max step**.
   - **Any** PR titled `[DRY RUN]…` or labelled `do-not-merge` fails.
   - The checker always runs as it is on the **base** branch, never the PR's copy.
   - Don't give the App the `workflows` permission. That is what stops its token from editing the guard workflow.

---

## Step 5 — Render the manifests → Config Sync

```bash
python3 setup/05-render-k8s.py      # → k8s/rendered/{serviceaccount,cronjob,networkpolicy,configmap-policy}.yaml
```

- Commit `k8s/rendered/*` to your Config Sync repo, under the `NAMESPACE` path.
- **Keep `"mode": "report"`** for the first in-cluster run.
- The CronJob uses `podFailurePolicy` (Kubernetes 1.26+): exits 2–5 fail the Job at once instead of running it twice.
- In report mode the job reads the tfvars file from GitHub (the App can read) but never writes.
- Check the NetworkPolicy against your datapath and egress design:
  - **[Verify]** the metadata-server rule (169.254.169.254:80 and 169.254.169.252:988).
  - Set `GOOGLE_APIS_CIDR` to private or restricted VIP.
  - Set `EGRESS_PROXY_CIDR` (and its port) for GitHub.
- The pod must pass Gatekeeper in deny mode. It runs non-root, with a read-only root filesystem, all capabilities dropped and no token mounted.

---

## Step 6 — Run once in the cluster

```bash
./setup/06-run-once-in-cluster.sh     # creates a Job from the CronJob and follows its logs
```

**Gate:** the same `target_excl_stockout` as Step 3b, and `token_source` in the log is `metadata` (Workload Identity).

- `report.json` stays in the pod. The **`sizer result`** log line carries the same numbers into Cloud Logging.

---

## Step 7 — Dry-run (1–2 weeks), then live

```bash
# config/policy.json: "mode": "dry-run"   → "[DRY RUN]" draft PRs labelled do-not-merge
python3 setup/05-render-k8s.py && # commit configmap-policy.yaml via Config Sync
./setup/06-run-once-in-cluster.sh
```

- **During dry-run:** review each draft PR against your manual numbers. Leave it unmerged; the guard fails it anyway.
  - The bot keeps **one** PR per cluster and pushes only when the proposed file changes.
  - It closes the PR, with a comment, when no increase is needed any more.
- **Go live:** set `"mode": "live"`, re-render, then commit.
  - The open dry-run PR is closed and a normal PR replaces it.
  - A human still approves and merges every PR, and HCP Terraform applies it.
- **After merge:** the next run compares the branch with Compute Engine, key by key.
  - Not applied yet, or the apply failed → that key is **held** (`PENDING_APPLY`, exit 3). Nothing is stacked on it, but the other keys are still sized.
  - If a key stays held for more than a day, check the HCP run, for example for `RESERVATION_CAPACITY_LIMIT_EXCEEDED` or `ZONE_RESOURCE_POOL_EXHAUSTED`.
  - **Stock-out:** open a human PR that sets the key to its `step_back_to` value (live + half the missing VMs), from the hold message or the PR body. The guard blocks decreases in bot PRs, so stepping back stays a human decision in the POC.
  - Live **above** the branch (`DRIFT`, changed outside Terraform) holds **every** key and closes the open bot PR, because any apply would shrink that reservation. Reconcile the file first.

---

## Reference

### Exit codes (alert on 2, 4, 5; on 3 only if it repeats; the Job fails at once on 2–5)

| Code | Meaning | What happened |
|---|---|---|
| 0 | OK | No change (open bot PR closed), a change proposed (report mode), or a PR raised/updated |
| 2 | **Fail-safe** | API error, coverage < `minDataCoverage`, empty series, machine-type mismatch. **Nothing written**; an open bot PR is left as it is. |
| 3 | **Hold** | At least one key held: branch > live, reservation missing or not READY. **Held keys keep their branch value; the other keys are still proposed.** Live > branch (`DRIFT`): **nothing written** and the open bot PR is closed. |
| 4 | Config | Bad policy or env, unknown pool, missing GitHub settings for PR modes |
| 5 | PR failed | GitHub API error after the numbers were computed |

### Row statuses in `report.json`

| Status | Meaning |
|---|---|
| `INCREASE` | The PR raises this key, clamped to max step |
| `OK` | Already at target |
| `COULD_REDUCE` | Advisory only, never written |
| `NO_POOLS_IN_SCOPE` | Key for another region or machine type, left alone |
| `COMMITMENT_ATTACHED` | Skipped, because a commitment-attached reservation needs the replace process |
| `NOT_CONSUMABLE` | No in-scope pool can use this reservation (affinity, Spot, or `specificReservationRequired`). Left unchanged. |
| `PENDING_APPLY` / `NOT_FOUND` / `NOT_READY` | Hold reasons for **that key** (`PENDING_APPLY` rows carry `step_back_to`) |
| `DRIFT` | Hold for **every key**: live above the branch |
| `NO_DATA` / `MISMATCH` | Fail-safe reasons |

### Policy (`config/policy.json`)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Kill switch. `false` exits 0 without doing anything. |
| `mode` | `report` | `report` (no PR) · `dry-run` (draft + do-not-merge) · `live` (normal PR) |
| `pools` | — | In-scope node pool **names** |
| `windowDays` / `bufferPct` | 7 / 10 | Same as today's manual rule |
| `surgeHeadroom` | `auto` | Largest `maxSurge` of the pools sharing a key, added **once** per zone (pools upgrade one at a time) |
| `stockoutAddPerDay` / `stockoutCapPerKey` | 1 / 2 | +1 per distinct stockout day in that zone, capped |
| `stockoutAmbiguous` | `skip` | Stockout whose error doesn't name the MIG and whose scale-up touched several zones: `skip` (report only) · `all-zones` (count it in each zone) |
| `maxStepPerKeyPerPR` | 3 | Clamp per key per PR (must match `SIZER_MAX_STEP`) |
| `minDataCoverage` | 0.9 | Below this, fail-safe |
| `historySource` | `node-metric` | `mig` = `compute.googleapis.com/instance_group/size`. **[Verify]** coverage in preflight before switching. |
| `excludeOperationTypes` | `["UPGRADE_NODES"]` | GKE operations whose surge nodes are excluded from peaks |
| `machineTypeAliases` | `{}` | Use when the pool's machine type string ≠ the reservation's (custom types) |

### Break-glass

- **Fastest:** **suspend the GitHub App installation** (org settings → Installed GitHub Apps). This stops all PR writes immediately.
- **Then:** set `"enabled": false` in the policy via Config Sync.
- Merged-but-wrong PR: revert it in Git, before HCP applies it if possible. `prevent_destroy` blocks any delete or replace.

---

## POC success criteria

- [ ] Step 1 plan shows **No changes**.
- [ ] Step 2 is within **±1 node per zone** of the manual numbers.
- [ ] Step 3b: impersonated run gives the same numbers; the bad-token run exits 2 with no file.
- [ ] Step 6: the in-cluster run uses `token_source=metadata`.
- [ ] **Tenant-noise drill:** a pod with an impossible `nodeSelector` in a test namespace does **not** change stockout events or targets. Delete the pod afterwards.
- [ ] **Upgrade drill:** a node pool upgrade inside the window appears in `upgrade_operations`, and the peak ignores the surge.
- [ ] One real PR merged + applied; the next run shows `OK`, not `PENDING_APPLY`.
- [ ] Security has reviewed: the 4-permission role, the conditional log view, the down-scoped App token, no K8s API access.
- [ ] **Guard drill:** a test PR from a `reservation-sizer/test` branch that also edits `ci/check_reservation_change.py` fails the guard.
- [ ] **Mode switch:** going from dry-run to live closes the draft PR and opens a normal one.

## Accepted POC risks (unchanged from v1.0)

- **GitHub key fetched in-cluster:** mitigated by the dedicated App, the down-scoped token and the CI guard. Proper fix: run the sizer outside the cluster.
- **ANY affinity:** the system pool isn't protected from user-pool bursts.
- **No N+1 or regional failover sizing.**
- **7-day window:** misses month-, quarter- and year-end peaks. Acceptable in non-prd only.
- **Named pools only:** auto-provisioned pools (today) and compute-class pools (planned) on the same machine type consume the same ANY reservations, but they are not sized.
  - Each row lists them in `out_of_scope_pools`.
  - Fix before compute classes go live: key demand by machine type and zone across all pools.
- **Stock-out step-back is manual:** after a failed apply, the key is held and the hold message suggests `step_back_to`. A human PR sets it, because bot PRs can't decrease.
- **Sum of per-pool peaks:** `target` can sit above the peak of the summed pools (Step 2). This is accepted as headroom.

## Still to verify in your environment

| Item | How it gets verified |
|---|---|
| GKE operations history covers `windowDays` | Preflight warns if not |
| `instance_group/size` metric (only if you switch `historySource`) | Preflight coverage |
| NetworkPolicy metadata-server rule for your datapath | Step 6 run |
| Conditional `viewAccessor` is enough for `entries.list` on the view | Step 3b impersonated run |
| Label API permission on PRs | Non-fatal; the guard also fails on the `[DRY RUN]` title |
| Stockout errors name the failing MIG in their parameters | Preflight (30 days): otherwise they appear as `stockout_ambiguous` |
| `podFailurePolicy` supported (Kubernetes 1.26+) | Step 6 run |

## Sources

- [View cluster autoscaler events and metrics](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/cluster-autoscaler-visibility): the events need Cloud Logging; `scale.up.error.out.of.resources`; noScaleUp is best-effort.
- [Cloud Monitoring quotas and limits](https://docs.cloud.google.com/monitoring/quotas): `kubernetes.io` and `compute.googleapis.com` metrics are kept for 24 months.
- [Configure log views on a log bucket](https://docs.cloud.google.com/logging/docs/logs-views): view filters; `viewAccessor` with an IAM condition per view.
- [Generating an installation access token for a GitHub App](https://docs.github.com/en/enterprise-cloud@latest/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation): token scoped to repositories and permissions.
- [About GKE cluster autoscaling](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler): unused reservations are prioritised in scale-up.
