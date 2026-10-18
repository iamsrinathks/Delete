## 2. Phase 0 - Pre-checks

Goal: confirm capacity, quota, policy and IAM exist before any reservation is created. **Gate:** every row in 2.3 and 2.4 is green.

### 2.1 Tooling and variables

```bash
# Tools: gcloud, kubectl, terraform >= 1.5, tflint, jq, conftest (optional)
gcloud components install gke-gcloud-auth-plugin beta

export OWNER_PROJECT="<-svp01-pvt-prj-id>"
export GKE_PROJECT="<-pre-cluster-project-id>"
export CONSUMER_PROJECT="<-svp01-pvt-prj-id>"
export DENIED_PROJECT="<non-listed-project-id>"
export REGION="europe-west2"
export ZONES="europe-west2-a europe-west2-b europe-west2-c"
export CLUSTER="<-cluster-name>"
export PREFIX="-pre-"

gcloud container clusters get-credentials "$CLUSTER" --region "$REGION" --project "$GKE_PROJECT"
```

### 2.2 Cluster baseline (capture before changes)

```bash
gcloud container clusters describe "$CLUSTER" --region "$REGION" --project "$GKE_PROJECT" \
  --format="yaml(currentMasterVersion,locations,autoscaling,nodePools[].name,nodePools[].config.machineType,nodePools[].config.reservationAffinity,nodePools[].autoscaling,nodePools[].upgradeSettings)" > evidence/00-cluster-baseline.yaml

kubectl get nodes -L topology.kubernetes.io/zone,node.kubernetes.io/instance-type,cloud.google.com/gke-nodepool > evidence/00-nodes-baseline.txt
gcloud compute reservations list --project "$OWNER_PROJECT" > evidence/00-reservations-baseline.txt
```

### 2.3 Zonal availability and quota

Run for every shape in Section 1. A shape missing in a zone means that zone cannot be reserved; record it and adjust the design.

```bash
for z in $ZONES; do
  echo "== $z"
  gcloud compute machine-types list --zones "$z" --project "$OWNER_PROJECT" \
    --filter="name~'^(n4-standard-8|g2-standard-8|g2-standard-96|g4-standard-48|a4-highgpu-8g)$'" \
    --format="value(name)"
  gcloud compute accelerator-types list --filter="zone:$z" --project "$OWNER_PROJECT" --format="value(name)"
done | tee evidence/00-zonal-availability.txt

# Custom shapes: confirm the series exists in the zone (custom shapes are not listed)
gcloud compute machine-types list --zones europe-west2-a --filter="name~'^(n2|n2d|n4)-standard-2$'" --project "$OWNER_PROJECT"

# Regional quota
gcloud compute regions describe "$REGION" --project "$OWNER_PROJECT" --format=json \
  | jq -r '.quotas[] | select(.metric|test("CPUS|N2_CPUS|N2D_CPUS|N4_CPUS|G2_CPUS|NVIDIA_L4|B200|LOCAL_SSD")) | "\(.metric)\t\(.usage)/\(.limit)"' \
  | tee evidence/00-quota.txt
```

| Check | Expected | Result |
| --- | --- | --- |
| N4 / N2 / N2D in all 3 zones | Present |  |
| `nvidia-l4` in all 3 zones | Present (note any missing zone) |  |
| G4 / RTX PRO 6000 in europe-west2 | Record yes/no; if no, G4 drops out of scope |  |
| A4 / B200 | Record yes/no; capacity via Google account team |  |
| Quota headroom >= sum of test reservations x 3 zones | Yes |  |

### 2.4 Org policy, IAM and approvals

```bash
# Shared reservation owner policy (needed for Phase 5)
gcloud resource-manager org-policies describe compute.sharedReservationsOwnerProjects \
  --project "$OWNER_PROJECT" --effective

# Who can create reservations in the owner project (should be the TFC SA only)
gcloud projects get-iam-policy "$OWNER_PROJECT" --format=json \
  | jq -r '.bindings[] | select(.role|test("compute.admin|compute.instanceAdmin")) | .role, .members[]'

# GKE service agents (needed for Phase 5)
GKE_PNUM=$(gcloud projects describe "$GKE_PROJECT" --format='value(projectNumber)')
echo "service-${GKE_PNUM}@container-engine-robot.iam.gserviceaccount.com"
```

- [ ] Org policy allows `$OWNER_PROJECT` as a shared-reservation owner
- [ ] TFC `prj` workspace SA has reservation create/update/delete in `$OWNER_PROJECT` only
- [ ] FinOps approval recorded (amount, duration, cost centre)
- [ ] A4: Google account team engaged for capacity type (future reservation / calendar mode / DWS) - **do not** assume an on-demand Terraform reservation works

## 3. Phase 1 - Static and policy tests

Goal: catch every bad input at `plan`, with zero cloud changes. **Gate:** all cases in 3.3 behave as expected and run in CI on every PR.

### 3.1 Lint and validate

```bash
cd lcp-gcp-gke-infra-services/modules/reservations
terraform fmt -check -recursive
terraform init -backend=false
terraform validate

cat > .tflint.hcl <<'EOF'
plugin "google" {
  enabled = true
  version = "0.30.0"
  source  = "github.com/terraform-linters/tflint-ruleset-google"
}
EOF
tflint --init && tflint
```

### 3.2 terraform test suite

Create `modules/reservations/tests/reservations.tftest.hcl`. All runs use `command = plan`, so nothing is created.

```hcl
mock_provider "google" {}

variables {
  project_id  = "tool01-svp01-pvt-prj"
  name_prefix = "tr1-pre-"
}

run "cpu_canary_plans" {
  command = plan
  variables {
    node_pool_reservations = {
      n4-std-8 = { enabled = true, zone = "europe-west2-a", instance_count = 1, machine_type = "n4-standard-8",
                   delete_after_duration = { seconds = 86400 } }
    }
  }
  assert {
    condition     = google_compute_reservation.node_pool["n4-std-8"].name == "tr1-pre-n4-std-8"
    error_message = "Name prefix not applied"
  }
  assert {
    condition     = google_compute_reservation.node_pool["n4-std-8"].specific_reservation_required == true
    error_message = "specific_reservation_required must default to true"
  }
}

run "disabled_is_skipped" {
  command = plan
  variables {
    node_pool_reservations = {
      off = { zone = "europe-west2-a", instance_count = 1, machine_type = "n4-standard-8" }
    }
  }
  assert {
    condition     = length(google_compute_reservation.node_pool) == 0
    error_message = "enabled defaults to false"
  }
}

run "gpu_and_shared" {
  command = plan
  variables {
    node_pool_reservations = {
      l4 = { enabled = true, zone = "europe-west2-b", instance_count = 1, machine_type = "g2-standard-8",
             accelerator_type = "nvidia-l4", accelerator_count = 1,
             share_type = "SPECIFIC_PROJECTS", shared_project_ids = ["caas01-svp01-pvt-prj"] }
    }
  }
}

# Negative cases: each must FAIL validation
run "neg_both_ttls" {
  command = plan
  variables {
    node_pool_reservations = { x = { enabled = true, zone = "europe-west2-a", instance_count = 1, machine_type = "n4-standard-8",
      delete_at_time = "2026-12-31T00:00:00Z", delete_after_duration = { seconds = 3600 } } }
  }
  expect_failures = [var.node_pool_reservations]
}

run "neg_machine_and_template" {
  command = plan
  variables {
    node_pool_reservations = { x = { enabled = true, zone = "europe-west2-a", instance_count = 1, machine_type = "n4-standard-8",
      source_instance_template = "projects/p/global/instanceTemplates/t" } }
  }
  expect_failures = [var.node_pool_reservations]
}

run "neg_accelerator_url" {
  command = plan
  variables {
    node_pool_reservations = { x = { enabled = true, zone = "europe-west2-a", instance_count = 1, machine_type = "g2-standard-8",
      accelerator_type = "projects/p/zones/europe-west2-a/acceleratorTypes/nvidia-l4", accelerator_count = 1 } }
  }
  expect_failures = [var.node_pool_reservations]
}
```

Run it: `terraform test` (add the remaining negatives from 3.3 the same way).

### 3.3 Negative validation cases

| ID | Input | Expected |
| --- | --- | --- |
| P1-01 | Both `delete_at_time` and `delete_after_duration` | Plan fails: mutually exclusive |
| P1-02 | `machine_type` and `source_instance_template` both set | Plan fails |
| P1-03 | Neither `machine_type` nor template | Plan fails |
| P1-04 | `accelerator_type` without `accelerator_count` | Plan fails |
| P1-05 | Accelerator as full URL | Plan fails |
| P1-06 | Key `L4_Pool` (uppercase/underscore) | Plan fails RFC1035 |
| P1-07 | `instance_count = 0` | Plan fails |
| P1-08 | `delete_at_time = "31/12/2026"` | Plan fails RFC3339 |
| P1-09 | `share_type = "SPECIFIC_PROJECTS"`, empty list | Plan fails |
| P1-10 | `service_share_type = "ALLOW"` | Plan fails enum |
| P1-11 | N4 shape with `local_ssds` | Plan passes; **apply fails** (N4 has no local SSD). Add a validation rule, then re-test |
| P1-12 | `seconds` passed as number | Plan shows string, no perpetual diff on second plan |

### 3.4 Sentinel policy in TFC (policy set on `prj` workspaces)

```hcl
# reservations.sentinel
import "tfplan/v2" as tfplan

allowed = {
  "pre": ["n4-standard-8","n4-standard-16","n4-standard-32","n2-standard-32","n2d-standard-32","n4-highmem-48",
          "n2-custom-32-32768","n2-custom-32-49152","n2-custom-32-65536","n2-custom-32-73728","n2-custom-32-90112",
          "n2-custom-64-147456","n2-custom-80-192512","n2d-custom-32-32768","n2d-custom-32-65536","n2d-custom-32-73728",
          "n2d-custom-32-90112","n4-custom-32-73728","g2-standard-8","g2-standard-16","g2-standard-96","a4-highgpu-8g"],
}
env = "pre"

res = filter tfplan.resource_changes as _, rc {
  rc.type is "google_compute_reservation" and rc.change.actions contains "create"
}

shape_ok = rule {
  all res as _, r {
    r.change.after.specific_reservation[0].instance_properties[0].machine_type in allowed[env]
  }
}
ttl_set = rule {
  all res as _, r {
    length(r.change.after.delete_after_duration else []) > 0 or (r.change.after.delete_at_time else "") is not ""
  }
}
gpu_specific = rule {
  all res as _, r {
    length(r.change.after.specific_reservation[0].instance_properties[0].guest_accelerators else []) == 0 or
    r.change.after.specific_reservation_required is true
  }
}
main = rule { shape_ok and ttl_set and gpu_specific }
```

- [ ] Policy set attached to tr1-pre `prj` workspace with enforcement `hard-mandatory`
- [ ] A plan with `n1-standard-2` is blocked
- [ ] A plan without a TTL is blocked
- [ ] A GPU plan with `specific_reservation_required = false` is blocked

## 4. Phase 2 - Reservation creation (canary)

Goal: create one reservation per class in all 3 zones, count 1, 24h TTL. **Gate:** all reservations READY, shapes exact, `inUseCount = 0`, second plan shows no changes.

### 4.1 tfvars for tr1-pre `prj` workspace

Reservation names must be **identical across zones** (same key, different zone) so one regional node pool can consume all three. Because the module keys by name, use a zone suffix in the key and set the final name per zone only if your module supports a `name` override; otherwise test the per-zone pattern in 4.2 first.

```hcl
project_id  = "<tool01-svp01-pvt-prj-id>"
name_prefix = "tr1-pre-"

node_pool_reservations = {
  t-n4-std8-a  = { enabled = true, zone = "europe-west2-a", instance_count = 1, machine_type = "n4-standard-8",
                   description = "TEST CPU canary, owner platform, CHG<id>", delete_after_duration = { seconds = 86400 } }
  t-n4-std8-b  = { enabled = true, zone = "europe-west2-b", instance_count = 1, machine_type = "n4-standard-8",
                   description = "TEST CPU canary, owner platform, CHG<id>", delete_after_duration = { seconds = 86400 } }
  t-n4-std8-c  = { enabled = true, zone = "europe-west2-c", instance_count = 1, machine_type = "n4-standard-8",
                   description = "TEST CPU canary, owner platform, CHG<id>", delete_after_duration = { seconds = 86400 } }

  t-n2c-32-64-a = { enabled = true, zone = "europe-west2-a", instance_count = 1, machine_type = "n2-custom-32-65536",
                    delete_after_duration = { seconds = 86400 } }

  t-l4-g2s8-a  = { enabled = true, zone = "europe-west2-a", instance_count = 2, machine_type = "g2-standard-8",
                   accelerator_type = "nvidia-l4", accelerator_count = 1, delete_after_duration = { seconds = 86400 } }
  t-l4-g2s8-b  = { enabled = true, zone = "europe-west2-b", instance_count = 2, machine_type = "g2-standard-8",
                   accelerator_type = "nvidia-l4", accelerator_count = 1, delete_after_duration = { seconds = 86400 } }
  t-l4-g2s8-c  = { enabled = true, zone = "europe-west2-c", instance_count = 2, machine_type = "g2-standard-8",
                   accelerator_type = "nvidia-l4", accelerator_count = 1, delete_after_duration = { seconds = 86400 } }
}
```

> **Design decision to prove (P2-00):** GKE `SPECIFIC_RESERVATION` takes a list of `values`. Test whether a regional node pool can list the three zone-specific names (`["tr1-pre-t-l4-g2s8-a","...-b","...-c"]`) and consume the right one per zone. If it cannot, add an optional `name` attribute to the module so the same name is used in every zone. Record the outcome; Phases 3-6 depend on it.

### 4.2 Run in TFC

1. Push the tfvars on a branch; open a PR. The speculative plan runs in TFC.
2. Check the plan: only `create` actions, count = number of entries, Sentinel passes.
3. Merge; confirm apply (auto-apply off for pre).
4. Save the run URL in the evidence log.

### 4.3 Verify in GCP

```bash
gcloud compute reservations list --project "$OWNER_PROJECT" --filter="name~^${PREFIX}t-" \
  --format="table(name,zone.basename(),specificReservation.count,specificReservation.inUseCount,specificReservation.instanceProperties.machineType,status)" \
  | tee evidence/02-reservations.txt

for z in $ZONES; do
  for r in $(gcloud compute reservations list --project "$OWNER_PROJECT" --filter="zone:$z AND name~^${PREFIX}t-" --format="value(name)"); do
    gcloud compute reservations describe "$r" --zone "$z" --project "$OWNER_PROJECT" --format=yaml > "evidence/02-$r.yaml"
  done
done
```

| ID | Check | Expected |
| --- | --- | --- |
| P2-01 | Status | `READY` for every reservation |
| P2-02 | Machine type, GPU type/count | Exactly as tfvars |
| P2-03 | `specificReservationRequired` | `true` |
| P2-04 | `deleteAfterDuration` / `deleteAtTime` | Present, 86400s |
| P2-05 | `inUseCount` | `0` |
| P2-06 | Second TFC plan (no code change) | `No changes` (proves no string/number drift on `seconds`) |
| P2-07 | Audit log `compute.reservations.insert` | Present, principal = TFC SA |
| P2-08 | Creation fails for a zone with no stock | Clear error captured; zone recorded as unavailable |

```bash
gcloud logging read 'protoPayload.methodName="v1.compute.reservations.insert"' \
  --project "$OWNER_PROJECT" --freshness=2h --format="table(timestamp,protoPayload.authenticationInfo.principalEmail,protoPayload.resourceName)"
```

## 5. Phase 3 - GKE consumption (platform canary pools)

Goal: prove each class is consumed by a tainted, platform-only node pool. **Gate:** every node consumes a reservation, `inUseCount` matches node count per zone, test pods run.

### 5.1 Canary node pools (TFC `gke` workspace)

```hcl
data "tfe_outputs" "prj" {
  organization = "<tfc-org>"
  workspace    = "<tr1-pre-tool01-prj-workspace>"
}

locals {
  rn = data.tfe_outputs.prj.nonsensitive_values.reservation_names
}

resource "google_container_node_pool" "rsv_test_l4" {
  name     = "rsv-test-l4"
  project  = var.gke_project
  cluster  = var.cluster_name
  location = "europe-west2"
  node_locations = ["europe-west2-a", "europe-west2-b", "europe-west2-c"]

  autoscaling {
    total_min_node_count = 0
    total_max_node_count = 6          # = sum of reserved count across zones
    location_policy      = "ANY"      # test BALANCED in Phase 6
  }

  upgrade_settings {
    max_surge       = 0
    max_unavailable = 1
  }

  node_config {
    machine_type = "g2-standard-8"
    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config { gpu_driver_version = "DEFAULT" }
    }
    labels = { "platform.abc/purpose" = "reservation-test" }
    taint {
      key    = "platform-test"
      value  = "true"
      effect = "NO_SCHEDULE"
    }
    reservation_affinity {
      consume_reservation_type = "SPECIFIC_RESERVATION"
      key                      = "compute.googleapis.com/reservation-name"
      values                   = [local.rn["t-l4-g2s8-a"], local.rn["t-l4-g2s8-b"], local.rn["t-l4-g2s8-c"]]
    }
  }
}
```

Repeat for `rsv-test-n4` (`n4-standard-8`, max 3) and `rsv-test-n2c` (`n2-custom-32-65536`, single zone `europe-west2-a`, max 1). Set up a **TFC run trigger** `prj -> gke` before applying.

### 5.2 Test namespace and manifests

```yaml
# 00-ns.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rsv-test
  labels:
    platform.abc/owner: platform
---
# 01-cpu-test.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-test
  namespace: rsv-test
spec:
  replicas: 3
  selector: { matchLabels: { app: cpu-test } }
  template:
    metadata: { labels: { app: cpu-test } }
    spec:
      nodeSelector:
        cloud.google.com/gke-nodepool: rsv-test-n4
      tolerations:
      - { key: platform-test, operator: Equal, value: "true", effect: NoSchedule }
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector: { matchLabels: { app: cpu-test } }
      containers:
      - name: pause
        image: registry.k8s.io/pause:3.9
        resources: { requests: { cpu: "6", memory: 20Gi } }   # forces 1 pod per n4-standard-8 node
---
# 02-gpu-test.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-smoke
  namespace: rsv-test
spec:
  parallelism: 3
  completions: 3
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-nodepool: rsv-test-l4
        cloud.google.com/gke-accelerator: nvidia-l4
      tolerations:
      - { key: platform-test, operator: Equal, value: "true", effect: NoSchedule }
      - { key: nvidia.com/gpu, operator: Exists, effect: NoSchedule }
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector: { matchLabels: { job-name: gpu-smoke } }
      containers:
      - name: smi
        image: nvidia/cuda:12.4.1-base-ubuntu22.04
        command: ["bash","-c","nvidia-smi && nvidia-smi --query-gpu=name,memory.total --format=csv"]
        resources: { limits: { nvidia.com/gpu: 1 } }
```

If your cluster only pulls from Artifact Registry, mirror `pause` and the CUDA image first.

### 5.3 Execute and verify

```bash
kubectl apply -f 00-ns.yaml -f 01-cpu-test.yaml -f 02-gpu-test.yaml
kubectl -n rsv-test get pods -o wide -w

# Nodes, zones, shapes
kubectl get nodes -l platform.abc/purpose=reservation-test \
  -L topology.kubernetes.io/zone,node.kubernetes.io/instance-type,cloud.google.com/gke-nodepool

# Reservation actually consumed by each VM
for n in $(kubectl get nodes -l platform.abc/purpose=reservation-test -o name | cut -d/ -f2); do
  z=$(kubectl get node $n -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
  gcloud compute instances describe $n --zone $z --project "$GKE_PROJECT" \
    --format="value(name,machineType.basename(),reservationAffinity.consumeReservationType,reservationAffinity.values)"
done

# inUseCount per reservation
gcloud compute reservations list --project "$OWNER_PROJECT" --filter="name~^${PREFIX}t-" \
  --format="table(name,zone.basename(),specificReservation.count,specificReservation.inUseCount)"

# GPU output and allocatable
kubectl -n rsv-test logs job/gpu-smoke --all-containers
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-l4 \
  -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
```

| ID | Check | Expected |
| --- | --- | --- |
| P3-01 | CPU pods | 3 Running, one per zone |
| P3-02 | `inUseCount` for `t-n4-std8-*` | 1 in each zone |
| P3-03 | VM `reservationAffinity` | `SPECIFIC_RESERVATION` + reservation name |
| P3-04 | GPU Job | 3/3 Completed, `nvidia-smi` shows NVIDIA L4 |
| P3-05 | `nvidia.com/gpu` allocatable | 1 per g2-standard-8 node |
| P3-06 | Custom N2 node instance type | Exactly `n2-custom-32-65536` |
| P3-07 | Repeat P3-04 with `g2-standard-96` | 8 GPUs allocatable per node |
| P3-08 | Time from pod pending to node Ready | Recorded (baseline for SLO) |
| P3-09 | No tenant pod on canary nodes | `kubectl get pods -A --field-selector spec.nodeName=<node>` shows only `rsv-test` and system pods |

## 6. Phase 4 - Negative and failure tests

Goal: know exactly how things fail, and capture the real error text for the runbook. **Gate:** every case behaves as expected and its error message is in the evidence log.

### 6.1 Shape mismatch (P4-01)

Create a temporary pool `rsv-test-mismatch` with `machine_type = "g2-standard-16"` but affinity to the `g2-standard-8` reservations, max 1.

```bash
kubectl -n rsv-test run mismatch --image=registry.k8s.io/pause:3.9 \
  --overrides='{"spec":{"nodeSelector":{"cloud.google.com/gke-nodepool":"rsv-test-mismatch"},"tolerations":[{"key":"platform-test","operator":"Exists"},{"key":"nvidia.com/gpu","operator":"Exists"}]}}'
kubectl -n rsv-test describe pod mismatch | sed -n '/Events/,$p'
gcloud logging read 'resource.type="k8s_cluster" AND jsonPayload.noDecisionStatus:* OR jsonPayload.resultInfo:*' \
  --project "$GKE_PROJECT" --freshness=30m --limit=20
gcloud logging read 'protoPayload.methodName:"instances.insert" AND severity>=ERROR' --project "$GKE_PROJECT" --freshness=30m
```

Expected: node never created, error references the reservation; **no fallback to on-demand**.

### 6.2 Exhaustion (P4-02)

```bash
kubectl -n rsv-test scale deploy/cpu-test --replicas=4   # reserved = 3 (1 per zone)
kubectl -n rsv-test get pods -o wide
kubectl -n rsv-test get events --field-selector reason=NotTriggerScaleUp,reason=FailedScheduling
```

Expected: 4th pod Pending, autoscaler scale-up fails for the reservation, no extra on-demand VM. Record the time until the autoscaler gives up and its backoff.

### 6.3 Theft test (P4-03)

```bash
# Plain VM, same shape, no affinity (defaults to ANY_RESERVATION)
gcloud compute instances create rsv-theft-test --zone europe-west2-a --project "$OWNER_PROJECT" \
  --machine-type n4-standard-8 --image-family debian-12 --image-project debian-cloud --no-address
gcloud compute reservations describe ${PREFIX}t-n4-std8-a --zone europe-west2-a --project "$OWNER_PROJECT" \
  --format="value(specificReservation.inUseCount)"
gcloud compute instances delete rsv-theft-test --zone europe-west2-a --project "$OWNER_PROJECT" --quiet
```

Expected: `inUseCount` unchanged, which proves `specific_reservation_required = true` works. Requires a network/subnet the test VM is allowed to use; get a security exception if VM creation outside GKE is blocked by org policy.

### 6.4 Explicit affinity from a plain VM (P4-04)

```bash
gcloud compute instances create rsv-explicit-test --zone europe-west2-a --project "$OWNER_PROJECT" \
  --machine-type n4-standard-8 --image-family debian-12 --image-project debian-cloud --no-address \
  --reservation-affinity=specific --reservation=${PREFIX}t-n4-std8-a
```

Expected: succeeds only if the slot is free. This is how a person outside the platform could consume capacity, so restrict `compute.instances.create` in the owner project.

### 6.5 Zone drain (P4-05)

```bash
Z=europe-west2-a
for n in $(kubectl get nodes -l platform.abc/purpose=reservation-test,topology.kubernetes.io/zone=$Z -o name); do
  kubectl cordon $n; kubectl drain $n --ignore-daemonsets --delete-emptydir-data --timeout=5m
done
kubectl -n rsv-test get pods -o wide
# restore
kubectl get nodes -l topology.kubernetes.io/zone=$Z -o name | xargs -n1 kubectl uncordon
```

Expected: with `DoNotSchedule` spread and no headroom in b/c, evicted pods stay **Pending**. Document this; it feeds the capacity rule in Section 8.

### 6.6 Node repair / VM deletion (P4-06)

```bash
NODE=$(kubectl get nodes -l cloud.google.com/gke-nodepool=rsv-test-n4 -o name | head -1 | cut -d/ -f2)
Z=$(kubectl get node $NODE -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
gcloud compute instances delete $NODE --zone $Z --project "$GKE_PROJECT" --quiet
kubectl get nodes -l cloud.google.com/gke-nodepool=rsv-test-n4 -w
```

Expected: MIG recreates the VM inside the same reservation; `inUseCount` drops then returns. Record recovery time.

| ID | Result | Error text captured |
| --- | --- | --- |
| P4-01 Shape mismatch |  |  |
| P4-02 Exhaustion |  |  |
| P4-03 Theft |  |  |
| P4-04 Explicit affinity VM |  |  |
| P4-05 Zone drain |  |  |
| P4-06 Node repair |  |  |

## 7. Phase 5 - Shared project reservations

Goal: prove a reservation owned in `$OWNER_PROJECT` is consumed only by listed consumer projects, with correct billing and safe change behaviour. **Gate:** P5-01 to P5-10 pass; IAM and org policy documented.

### 7.1 Setup

Add a shared reservation set (3 zones) to the `prj` tfvars:

```hcl
  t-shared-l4-a = { enabled = true, zone = "europe-west2-a", instance_count = 1, machine_type = "g2-standard-8",
                    accelerator_type = "nvidia-l4", accelerator_count = 1,
                    share_type = "SPECIFIC_PROJECTS", shared_project_ids = ["<GKE_PROJECT>", "<CONSUMER_PROJECT>"],
                    delete_after_duration = { seconds = 86400 } }
  # repeat -b and -c
```

Grant the consumer GKE service agent read access to reservations in the owner project. Confirm the exact permission against current GKE documentation during this test and record it.

```bash
gcloud iam roles create reservationReader --project "$OWNER_PROJECT" \
  --title="Reservation reader" --permissions=compute.reservations.get,compute.reservations.list

CPNUM=$(gcloud projects describe "$CONSUMER_PROJECT" --format='value(projectNumber)')
gcloud projects add-iam-policy-binding "$OWNER_PROJECT" \
  --member="serviceAccount:service-${CPNUM}@container-engine-robot.iam.gserviceaccount.com" \
  --role="projects/${OWNER_PROJECT}/roles/reservationReader"
```

Codify this binding in Terraform after the test; manual grants are for discovery only.

Consumer node pool affinity uses the **full path** output:

```hcl
values = [
  data.tfe_outputs.prj.nonsensitive_values.shared_reservation_paths["t-shared-l4-a"],
  data.tfe_outputs.prj.nonsensitive_values.shared_reservation_paths["t-shared-l4-b"],
  data.tfe_outputs.prj.nonsensitive_values.shared_reservation_paths["t-shared-l4-c"],
]
```

### 7.2 Verify sharing config

```bash
gcloud compute reservations describe ${PREFIX}t-shared-l4-a --zone europe-west2-a --project "$OWNER_PROJECT" \
  --format="yaml(shareSettings,reservationSharingPolicy,specificReservationRequired)"
```

### 7.3 Test cases

| ID | Step | Expected |
| --- | --- | --- |
| P5-01 | Consumer pool with full-path affinity, run `gpu-smoke` | Nodes Ready; `inUseCount` increments in **owner** project |
| P5-02 | Consumer pool with bare name only | Fails (looks in local project); capture error |
| P5-03 | Remove the IAM binding, scale up | Fails; capture error (often reads as "not found") |
| P5-04 | Pool in `$DENIED_PROJECT` with full path | Denied |
| P5-05 | Listed consumer using `ANY_RESERVATION` | Does **not** consume (specific required) |
| P5-06 | Two listed consumers race for last slot | First wins; second Pending; document as fairness rule |
| P5-07 | Add a project to `shared_project_ids` | TFC plan = in-place `update`, not replace |
| P5-08 | Remove a consumer while its nodes run | Running VMs keep running; scale-up/upgrade/repair fail. Capture both |
| P5-09 | Billing export after 24h | Idle reserved cost on owner; VM cost on consumer |
| P5-10 | `service_share_type = DISALLOW_ALL` | Vertex AI cannot consume; GKE unaffected |

P5-08 commands:

```bash
# after removing CONSUMER_PROJECT from shared_project_ids and applying in TFC
kubectl --context <consumer-cluster> -n rsv-test get pods -o wide        # still Running
kubectl --context <consumer-cluster> -n rsv-test scale job/gpu-smoke --replicas=2 2>/dev/null || true
gcloud container clusters resize <consumer-cluster> --node-pool rsv-test-l4 --num-nodes 2 --region "$REGION" --project "$CONSUMER_PROJECT"
```

P5-09 billing query (BigQuery detailed export):

```sql
SELECT project.id, sku.description, SUM(cost) AS cost, SUM(usage.amount_in_pricing_units) AS usage
FROM `<billing_project>.<dataset>.gcp_billing_export_resource_v1_<ACCOUNT>`
WHERE usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY)
  AND project.id IN ('<OWNER_PROJECT>','<CONSUMER_PROJECT>')
  AND (LOWER(sku.description) LIKE '%reserv%' OR LOWER(sku.description) LIKE '%l4%' OR LOWER(sku.description) LIKE '%g2%')
GROUP BY 1,2 ORDER BY 1,3 DESC;
```

### 7.4 Guardrails to raise from this phase

- [ ] Sentinel: any removal from `shared_project_ids` needs a policy override (CAB-approved)
- [ ] IAM binding for consumer GKE service agents managed in Terraform
- [ ] Runbook entry for "reservation not found" = check IAM, full path, share list

## 8. Phase 6 - Topology spread and scheduling

Goal: show how tenant spread rules behave against fixed per-zone reserved capacity. **Gate:** matrix results recorded, `location_policy` chosen, tenant guidance written.

### 8.1 Test deployment (parameterised)

```yaml
# 10-spread-test.yaml  -- edit REPLICAS / WHEN / pool per case
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spread-test
  namespace: rsv-test
spec:
  replicas: 6                       # REPLICAS
  selector: { matchLabels: { app: spread-test } }
  template:
    metadata: { labels: { app: spread-test } }
    spec:
      nodeSelector: { cloud.google.com/gke-nodepool: rsv-test-n4 }
      tolerations:
      - { key: platform-test, operator: Equal, value: "true", effect: NoSchedule }
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule   # WHEN: DoNotSchedule | ScheduleAnyway
        labelSelector: { matchLabels: { app: spread-test } }
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector: { matchLabels: { app: spread-test } }
            topologyKey: kubernetes.io/hostname
      containers:
      - name: pause
        image: registry.k8s.io/pause:3.9
        resources: { requests: { cpu: "1", memory: 1Gi } }
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: spread-test, namespace: rsv-test }
spec:
  minAvailable: 4
  selector: { matchLabels: { app: spread-test } }
```

For each case, set reservation counts per zone in `prj` tfvars, apply, set pool `total_max_node_count` = total reserved, then:

```bash
kubectl apply -f 10-spread-test.yaml
sleep 600   # allow autoscaler time; or watch
kubectl -n rsv-test get pods -l app=spread-test -o wide --sort-by=.spec.nodeName
kubectl -n rsv-test get pods -l app=spread-test \
  -o jsonpath='{range .items[*]}{.status.phase}{"\t"}{.spec.nodeName}{"\n"}{end}' | sort | uniq -c
kubectl -n rsv-test get events --sort-by=.lastTimestamp | grep -Ei 'spread|FailedScheduling|TriggeredScaleUp|NotTriggerScaleUp'
gcloud compute reservations list --project "$OWNER_PROJECT" --filter="name~^${PREFIX}t-n4" \
  --format="table(name,zone.basename(),specificReservation.count,specificReservation.inUseCount)"
```

### 8.2 Test matrix

| ID | Reserved a/b/c | Spread | Replicas | location\_policy | Expected |
| --- | --- | --- | --- | --- | --- |
| P6-01 | 2/2/2 | DoNotSchedule | 6 | ANY | 6 Running, 2 per zone |
| P6-02 | 2/2/2 | DoNotSchedule | 7 | ANY | 1 Pending |
| P6-03 | 3/3/1 | DoNotSchedule | 6 | ANY | Pending once zone c is full (expect 2/2/1 + 1 Pending) |
| P6-04 | 3/3/1 | ScheduleAnyway | 6 | ANY | 6 Running, skewed |
| P6-05 | 3/3/1 | DoNotSchedule | 6 | BALANCED | Compare with P6-03; record autoscaler loops/backoff |
| P6-06 | 2/2/2 | DoNotSchedule + hostname anti-affinity | 6 | ANY | Exactly 1 pod per node |
| P6-07 | 2/2/2 | DoNotSchedule | 6, then drain zone a | ANY | 2 Pending (no headroom) |
| P6-08 | 3/3/3 | DoNotSchedule | 6, then drain zone a | ANY | Pods move only if skew allows; record result |
| P6-09 | 2/2/2 | DoNotSchedule | scale 6 -> 3 | ANY | Scale-down keeps skew <= 1; PDB honoured |

Capture for each: pod placement per zone, Pending count, autoscaler events, time to steady state.

### 8.3 Zonal PV pinning (P6-10)

```yaml
# 11-sts-pv.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata: { name: pv-test, namespace: rsv-test }
spec:
  serviceName: pv-test
  replicas: 3
  selector: { matchLabels: { app: pv-test } }
  template:
    metadata: { labels: { app: pv-test } }
    spec:
      nodeSelector: { cloud.google.com/gke-nodepool: rsv-test-n4 }
      tolerations: [{ key: platform-test, operator: Equal, value: "true", effect: NoSchedule }]
      topologySpreadConstraints:
      - { maxSkew: 1, topologyKey: topology.kubernetes.io/zone, whenUnsatisfiable: DoNotSchedule, labelSelector: { matchLabels: { app: pv-test } } }
      containers:
      - name: pause
        image: registry.k8s.io/pause:3.9
        volumeMounts: [{ name: data, mountPath: /data }]
  volumeClaimTemplates:
  - metadata: { name: data }
    spec:
      accessModes: [ReadWriteOnce]
      storageClassName: standard-rwo
      resources: { requests: { storage: 1Gi } }
```

```bash
kubectl apply -f 11-sts-pv.yaml
# Fill zone a with other pods so its reservation is full, then:
kubectl -n rsv-test delete pod pv-test-0      # assume pv-test-0 lives in zone a
kubectl -n rsv-test describe pod pv-test-0 | sed -n '/Events/,$p'
```

Expected: `pv-test-0` stays Pending (`volume node affinity conflict` / no node in zone). This is a tenant-facing risk for StatefulSets on reserved pools.

### 8.4 A4 / multi-GPU training placement (P6-11, only when A4 capacity exists)

- A4 jobs should stay in **one zone** (compact placement); cross-zone spread is wrong for them.
- Test: a JobSet or Job with `nodeSelector: topology.kubernetes.io/zone: <zone with A4 reservation>` and **no** zone spread; confirm all workers land on reserved A4 nodes in that zone.
- Negative: same Job with zone spread `DoNotSchedule` -> expect Pending in zones without A4; capture as tenant guidance.

### 8.5 Decisions to record

- [ ] `location_policy` for reserved pools (expected: `ANY`)
- [ ] Rule for tenants on reserved pools: `ScheduleAnyway` by default, `DoNotSchedule` only with approval
- [ ] Headroom rule for zone-loss tolerance (e.g. reserve +50% per zone for N+1) agreed with service owners
- [ ] Guidance for StatefulSets with zonal PVs

## 9. Phase 7 - Lifecycle and day-2 operations

Goal: prove changes, expiry, upgrades and maintenance do not cause outages. **Gate:** all cases pass; upgrade settings for reserved pools decided.

| ID | Action | How | Expected |
| --- | --- | --- | --- |
| P7-01 | Resize up | `instance_count` 1 -> 2 in tfvars, TFC plan/apply | Plan shows in-place `update`; count becomes 2 |
| P7-02 | Resize down below in-use | Set count < `inUseCount` | Record behaviour (API error or accepted); running VMs unaffected |
| P7-03 | ForceNew guard | Change `machine_type` | Plan **errors** on `prevent_destroy` |
| P7-04 | Disable entry | `enabled = false` | Plan errors on `prevent_destroy` (expected); retire via documented procedure (P7-11) |
| P7-05 | TTL expiry | Let 24h `delete_after_duration` fire | Reservation gone; running nodes stay; new nodes fail |
| P7-06 | Drift after TTL | Next TFC plan | Plan wants to re-create; decide: remove entry or re-apply |
| P7-07 | GKE node upgrade, surge 0 | Upgrade canary pool | Nodes replaced one by one inside reservation; no extra capacity needed |
| P7-08 | GKE node upgrade, surge 1 | `max_surge = 1`, reservation full | Surge node fails; upgrade stalls. Prove you need +1 slot per zone |
| P7-09 | GPU host maintenance | Simulate | GPU VM stops and restarts; pod reschedules; node rejoins reservation |
| P7-10 | Run trigger | Apply `prj` change | `gke` workspace plan queued automatically; `tfe_outputs` values correct |
| P7-11 | Planned retirement | Procedure below | No unplanned pod loss |

### 9.1 Commands

```bash
# P7-07 / P7-08 upgrade
gcloud container node-pools update rsv-test-l4 --cluster "$CLUSTER" --region "$REGION" --project "$GKE_PROJECT" \
  --max-surge-upgrade=0 --max-unavailable-upgrade=1
gcloud container clusters upgrade "$CLUSTER" --node-pool rsv-test-l4 --region "$REGION" --project "$GKE_PROJECT" \
  --cluster-version $(gcloud container clusters describe "$CLUSTER" --region "$REGION" --project "$GKE_PROJECT" --format='value(currentMasterVersion)') --quiet
kubectl get nodes -l cloud.google.com/gke-nodepool=rsv-test-l4 -w

# Repeat with surge 1 for P7-08
gcloud container node-pools update rsv-test-l4 --cluster "$CLUSTER" --region "$REGION" --project "$GKE_PROJECT" \
  --max-surge-upgrade=1 --max-unavailable-upgrade=0

# P7-09 host maintenance simulation on one GPU node
NODE=$(kubectl get nodes -l cloud.google.com/gke-nodepool=rsv-test-l4 -o name | head -1 | cut -d/ -f2)
Z=$(kubectl get node $NODE -o jsonpath='{.metadata.labels.topology\.kubernetes\.io/zone}')
gcloud compute instances simulate-maintenance-event $NODE --zone $Z --project "$GKE_PROJECT"
kubectl get node $NODE -w

# P7-05 TTL check
gcloud compute reservations list --project "$OWNER_PROJECT" --filter="name~^${PREFIX}t-"
```

### 9.2 Planned retirement procedure (P7-11)

1. Scale tenant workloads or move them to a non-reserved pool.
2. Remove `reservation_affinity` from the node pool (or set `NO_RESERVATION`) in the `gke` workspace; apply. New nodes become on-demand.
3. Confirm `inUseCount = 0`.
4. In the `prj` workspace, temporarily remove `prevent_destroy` via an approved PR (or `terraform state rm` then delete with gcloud, per change control).
5. Apply; verify reservation deleted and audit log entry present.
6. Never delete the reservation before step 2.

### 9.3 Decisions to record

- [ ] Reserved pools use `max_surge = 0, max_unavailable = 1`, **or** `instance_count` = max nodes + 1 per zone
- [ ] Maintenance window and exclusion policy for GPU pools
- [ ] TTL policy: pre/int must have TTL; prd steady capacity none

### 9.4 Modifying and deleting reservations that are in use

Goal: prove which reservation changes are in-place, which recreate the reservation, and what happens to running nodes and the node pool in each case. **Gate:** every row below has a recorded result. Use the `t-n4-std8-*` and `t-l4-g2s8-*` reservations with the canary pools **running** (`inUseCount` > 0).

**Expected behaviour (to confirm)**

| Change | Reservation | Running nodes | Node pool (gke workspace) |
| --- | --- | --- | --- |
| `instance_count` up | In-place update | Unaffected | No change |
| `instance_count` down, still >= in-use | In-place update | Unaffected | No change |
| `instance_count` down below in-use | Record: rejected or accepted | Must stay Running | No change |
| `shared_project_ids` add/remove | In-place update | Unaffected (removed consumer: new nodes fail) | No change |
| TTL (`delete_at_time` / `delete_after_duration`) change | Record: in-place or replace | Unaffected | No change |
| Shape change (machine type, GPU, local SSD, min CPU, template) | Replace (blocked by `prevent_destroy`) | Record behaviour if forced | No change, unless name changes |
| `specific_reservation_required` toggle | Record: in-place or replace | Unaffected | No change |
| Reservation **name** change | Replace | Unaffected | `reservation_affinity.values` changes: record if node pool is updated in place or **recreated** |
| Delete reservation while consumed | Deleted | Keep running, billed on-demand, no capacity guarantee | New nodes (scale-up, upgrade, repair) fail with `SPECIFIC_RESERVATION` |

Key point: the reservation and the node pool are in **different TFC workspaces**. A reservation change never touches the node pool resource. Only a change to the node pool's `reservation_affinity` can recreate the pool.

**Test steps**

```bash
# Baseline before each case
kubectl get nodes -l platform.abc/purpose=reservation-test -o wide > evidence/94-before.txt
gcloud compute reservations list --project "$OWNER_PROJECT" --filter="name~^${PREFIX}t-" \
  --format="table(name,zone.basename(),specificReservation.count,specificReservation.inUseCount)" >> evidence/94-before.txt
kubectl -n rsv-test get pods -o wide >> evidence/94-before.txt
```

| ID | Step | Record |
| --- | --- | --- |
| P7-12 | Change `instance_count` 2 -> 3 on `t-l4-g2s8-a`; check TFC plan action | `update` vs `replace`; node uptime unchanged |
| P7-13 | Change `instance_count` to 1 while `inUseCount` = 2 | API response; node state after 15 min |
| P7-14 | Change `delete_after_duration` 86400 -> 172800 | Plan action |
| P7-15 | Toggle `specific_reservation_required` true -> false | Plan action |
| P7-16 | Change `machine_type` g2-standard-8 -> g2-standard-16 | Plan blocked by `prevent_destroy` (expected) |
| P7-17 | In an approved test PR, remove `prevent_destroy` and repeat P7-16 | Destroy-then-create sequence; do running nodes stay up? Does `inUseCount` on the new reservation count them? |
| P7-18 | Rename a reservation key, update node pool affinity in `gke` workspace | `gke` plan: node pool `update` in place vs `forces replacement`; **stop before apply if it says replace** and record it |
| P7-19 | Delete a consumed reservation directly | See commands below |
| P7-20 | After P7-19, scale pool +1 and run a node upgrade | Scale-up and surge fail; capture errors |
| P7-21 | Recreate the same reservation name after P7-19 | Do existing running nodes get counted in `inUseCount` again, or only new nodes? |

P7-19 / P7-21 commands:

```bash
R=${PREFIX}t-n4-std8-a; Z=europe-west2-a
gcloud compute reservations describe $R --zone $Z --project "$OWNER_PROJECT" --format="value(specificReservation.inUseCount)"
gcloud compute reservations delete $R --zone $Z --project "$OWNER_PROJECT" --quiet
kubectl get nodes -l cloud.google.com/gke-nodepool=rsv-test-n4,topology.kubernetes.io/zone=$Z -o wide   # still Ready?
kubectl -n rsv-test get pods -o wide                                                                      # still Running?

# P7-20
kubectl -n rsv-test scale deploy/cpu-test --replicas=4
kubectl -n rsv-test get events --sort-by=.lastTimestamp | tail -20

# P7-21: re-apply the prj workspace (recreates the reservation), then
gcloud compute reservations describe $R --zone $Z --project "$OWNER_PROJECT" --format="value(specificReservation.inUseCount)"
```

Also record: whether a reservation attached to a **commitment (CUD)** can be deleted (expected: not until detached), and how deletion of a **shared** reservation affects consumer projects (same as P7-19, in the consumer cluster).

### 9.5 Confirmed by Google documentation (Oct 2026)

These rules are now documented; tests P7-12 to P7-21 confirm them in tr1-pre rather than discover them.

| Topic | Documented behaviour | Impact on plan |
| --- | --- | --- |
| Regional node pool | Needs a specific reservation with **the same name and properties in every zone** of the pool | P2-00 resolved: module now has `name` (shared) + per-zone map key |
| Default affinity | `ANY` is the Compute Engine default | Omit `reservation_affinity` on non-reserved pools; adding it forces node pool replacement |
| Pool size | Specific-affinity pool is capped at reservation capacity for life; upgrades and autoscale need spare slots | Keep `max_surge = 0` or resize reservation |
| GPU pools | Set `node_locations` to zones with reserved capacity | Use module output `reservation_zones` |
| Uneven zones | `location_policy = ANY` follows unused reservations | Default for reserved pools |
| In-place changes | Only count, share type, consumer projects, Vertex sharing policy, auto-delete | Shape or name change = new reservation (blocked by `prevent_destroy`) |
| Shrink count | Cannot go below instances consuming a **specific** reservation | P7-13 expected: API rejects |
| Remove consumer project | Only allowed when no instances in that project consume it | P5-08 expected: API rejects while nodes run |
| Delete | A **specific** reservation can only be deleted when nothing consumes it | P7-19 expected: delete rejected while nodes run |
| Stopped VMs | A stopped/suspended VM that targeted a deleted reservation can only restart after a reservation with the same name, zone and properties is recreated | Add to runbook |
| Shared IAM | Consumer GKE service agent needs a custom role with `compute.reservations.list` in the owner project | Update 7.1: drop `.get` |
| Shared limits | Up to 100 consumer projects, same org; only owner project can modify or delete | Module validation added |
| CUD-attached (GPU, most local SSD) | Cannot delete, resize or enable auto-delete; auto-deleted when commitment expires | Module flag `commitment_attached` blocks TTL |
| Compute flexible CUDs | Cover G2/G4 GPUs without attached reservations, but give no capacity guarantee | FinOps decision input |
| Shared + CUD | Consumers only get the owner's CUD if CUD sharing is on and projects share the billing account | Check in P5-09 |
| Auto-delete flags in gcloud | `--delete-at-time` / `--delete-after-duration` are under `gcloud beta` | Use Terraform or `gcloud beta` in manual tests |

Sources: [Consuming reservations in GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/consuming-reservations) · [Single-project](https://docs.cloud.google.com/compute/docs/instances/reservations-single-project) · [Shared](https://docs.cloud.google.com/compute/docs/instances/reservations-shared) · [With commitments](https://docs.cloud.google.com/compute/docs/instances/reservations-with-commitments) · [Modify](https://docs.cloud.google.com/compute/docs/instances/reservations-modify) · [Delete](https://docs.cloud.google.com/compute/docs/instances/reservations-delete)

## 10. Phase 8 - Multi-tenant governance

Goal: only approved tenants land on reserved pools, and no tenant can exhaust shared reserved capacity. **Gate:** Gatekeeper constraint enforced (dryrun first, then deny), quota blocks over-requests. Deliver these via your Config Sync repo, not `kubectl apply`, outside of the test namespace.

### 10.1 Reserved pools taint (production pattern)

Reserved pools in prd will carry `platform.abc/reserved=<pool>:NoSchedule`. Only approved namespaces may tolerate it.

### 10.2 Gatekeeper constraint

```yaml
# 20-template.yaml
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: k8sreservedpooltoleration
spec:
  crd:
    spec:
      names: { kind: K8sReservedPoolToleration }
      validation:
        openAPIV3Schema:
          type: object
          properties:
            taintKey: { type: string }
            allowedNamespaces: { type: array, items: { type: string } }
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package k8sreservedpooltoleration
      tolerates[t] {
        t := input.review.object.spec.tolerations[_]
        t.key == input.parameters.taintKey
      }
      tolerates[t] {
        t := input.review.object.spec.template.spec.tolerations[_]
        t.key == input.parameters.taintKey
      }
      violation[{"msg": msg}] {
        count(tolerates) > 0
        ns := input.review.object.metadata.namespace
        not allowed(ns)
        msg := sprintf("namespace %v is not approved for reserved pools (%v)", [ns, input.parameters.taintKey])
      }
      allowed(ns) { ns == input.parameters.allowedNamespaces[_] }
---
# 21-constraint.yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sReservedPoolToleration
metadata:
  name: reserved-pool-toleration
spec:
  enforcementAction: dryrun     # then warn, then deny
  match:
    kinds:
    - { apiGroups: [""], kinds: ["Pod"] }
    - { apiGroups: ["apps"], kinds: ["Deployment","StatefulSet","DaemonSet"] }
    - { apiGroups: ["batch"], kinds: ["Job"] }
  parameters:
    taintKey: platform-test
    allowedNamespaces: ["rsv-test"]
```

```bash
kubectl apply -f 20-template.yaml && kubectl apply -f 21-constraint.yaml
kubectl create ns rsv-intruder
kubectl -n rsv-intruder run intruder --image=registry.k8s.io/pause:3.9 \
  --overrides='{"spec":{"tolerations":[{"key":"platform-test","operator":"Exists"}]}}'
kubectl get k8sreservedpooltoleration reserved-pool-toleration -o jsonpath='{.status.violations}' | jq
# switch to deny and retry
kubectl patch k8sreservedpooltoleration reserved-pool-toleration --type merge -p '{"spec":{"enforcementAction":"deny"}}'
kubectl -n rsv-intruder run intruder2 --image=registry.k8s.io/pause:3.9 \
  --overrides='{"spec":{"tolerations":[{"key":"platform-test","operator":"Exists"}]}}'
```

Also test a second constraint, in dryrun, that flags `whenUnsatisfiable: DoNotSchedule` on reserved pools unless the namespace has label `platform.abc/strict-spread-approved=true` (decision from Section 8).

### 10.3 GPU quota per namespace

```yaml
# 22-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata: { name: gpu-quota, namespace: rsv-test }
spec:
  hard:
    requests.nvidia.com/gpu: "2"
```

```bash
kubectl apply -f 22-quota.yaml
kubectl -n rsv-test delete job gpu-smoke --ignore-not-found
kubectl -n rsv-test apply -f 02-gpu-test.yaml     # parallelism 3 -> 3rd pod rejected
kubectl -n rsv-test get events | grep -i 'exceeded quota'
```

### 10.4 Test cases

| ID | Check | Expected |
| --- | --- | --- |
| P8-01 | Intruder namespace tolerates reserved taint (dryrun) | Violation reported |
| P8-02 | Same in deny mode | Admission rejected with clear message |
| P8-03 | Approved namespace | Admitted |
| P8-04 | GPU request over quota | Rejected by ResourceQuota |
| P8-05 | Existing tenant workloads | No new violations in audit (run constraint in dryrun 7 days first) |
| P8-06 | Reserved node labels/taints | Visible on nodes; documented for tenants |
| P8-07 | Platform-only node pool changes | Only platform group has GKE node pool IAM |

```bash
kubectl delete ns rsv-intruder
```

## 11. Phase 9 - Observability and FinOps

Goal: on-call sees waste and exhaustion before tenants do, and cost lands on the right cost centre. **Gate:** alerts fire in test, dashboards live, billing and audit verified.

### 11.1 Utilisation exporter (until a native metric is confirmed)

Reservation `inUseCount / count` is the key signal. If no native Cloud Monitoring metric covers it in your setup, run a small CronJob that writes a custom metric:

```yaml
# 30-rsv-exporter.yaml (runs with Workload Identity SA that has compute.reservations.list + monitoring.metricWriter)
apiVersion: batch/v1
kind: CronJob
metadata: { name: rsv-util-exporter, namespace: platform-observability }
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: rsv-exporter
          restartPolicy: Never
          containers:
          - name: export
            image: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
            env: [{ name: OWNER_PROJECT, value: "<owner-project>" }]
            command: ["bash","-c"]
            args:
            - |
              set -e
              TOKEN=$(gcloud auth print-access-token)
              NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
              gcloud compute reservations list --project "$OWNER_PROJECT" \
                --format="csv[no-heading](name,zone.basename(),specificReservation.count,specificReservation.inUseCount)" |
              while IFS=, read -r n z c u; do
                u=${u:-0}
                curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
                  "https://monitoring.googleapis.com/v3/projects/$OWNER_PROJECT/timeSeries" -d "{\"timeSeries\":[{
                    \"metric\":{\"type\":\"custom.googleapis.com/reservation/utilisation\",\"labels\":{\"reservation\":\"$n\",\"zone\":\"$z\"}},
                    \"resource\":{\"type\":\"global\",\"labels\":{\"project_id\":\"$OWNER_PROJECT\"}},
                    \"points\":[{\"interval\":{\"endTime\":\"$NOW\"},\"value\":{\"doubleValue\":$(awk "BEGIN{print $u/$c}")}}]}]}"
              done
```

### 11.2 Alert policies

| Alert | Condition | Route |
| --- | --- | --- |
| Reservation waste | utilisation < 0.5 for 7 days | FinOps + platform (ticket) |
| Reservation exhausted | utilisation = 1.0 for 30 min | Platform on-call (warning) |
| Pending on reserved pool | Pods Pending > 10 min with `FailedScheduling` on reserved pool | Platform on-call |
| Scale-up blocked | Autoscaler `scale.up.error` events on reserved pool | Platform on-call |
| Reservation deleted | Audit log `reservations.delete` | Platform + security |
| Shared list changed | Audit log `reservations.update` with shareSettings | Platform + security |

Log-based metric for scale-up errors:

```bash
gcloud logging metrics create rsv_scaleup_errors --project "$GKE_PROJECT" \
  --description="Cluster autoscaler scale-up errors" \
  --log-filter='resource.type="k8s_cluster" AND logName:"cluster-autoscaler-visibility" AND jsonPayload.resultInfo.results.errorMsg.messageId:"scale.up.error"'
```

Audit log alert filter:

```text
protoPayload.serviceName="compute.googleapis.com"
protoPayload.methodName=~"reservations\.(delete|update|resize)"
```

### 11.3 Test cases

| ID | Trigger | Expected |
| --- | --- | --- |
| P9-01 | Exporter run | Metric visible in Metrics Explorer per reservation and zone |
| P9-02 | Exhaustion (reuse P4-02) | Exhausted + Pending alerts fire within SLA |
| P9-03 | Idle reservation (lower threshold to 1h for test) | Waste alert fires |
| P9-04 | Delete a test reservation (after P7-11) | Delete alert fires, reaches SIEM |
| P9-05 | Billing export | Reservation cost visible with cost-centre labels/description |
| P9-06 | Dashboard | Per class: reserved vs used vs pending pods, per zone |

### 11.4 Billing check

The reservation resource does not take labels in this module; cost attribution comes from the project and the `description`. Confirm FinOps can attribute it; if not, raise a follow-up to add labels where the provider supports them.

## 12. Phase 10 - Tenant pilot, rollback, cleanup and sign-off

### 12.1 Tenant pilot (1-2 weeks)

1. Pick 1-2 consumer teams with non-critical GPU or high-CPU workloads in pre.
2. Add their namespaces to the Gatekeeper `allowedNamespaces` and give them a GPU ResourceQuota.
3. Share the tenant guide: node selector, toleration, quota, `ScheduleAnyway` rule, "Pending when reservation is full" behaviour, StatefulSet/PV caveat.
4. Track daily: pod scheduling latency, Pending count, `FailedScheduling` events, utilisation, tenant feedback.
5. Exit: no P1/P2 incidents, tenants confirm workloads healthy.

### 12.2 Rollback (any phase)

| Situation | Action |
| --- | --- |
| Reserved pool cannot get nodes | Remove `reservation_affinity` (or `NO_RESERVATION`) in `gke` workspace and apply; nodes go on-demand |
| Wrong reservation shape | Do not change shape in place; create a new entry, move pool affinity, then retire old (P7-11) |
| Shared consumer broken | Re-add project to `shared_project_ids`; verify IAM binding |
| Gatekeeper blocks a tenant | Set constraint to `dryrun`, fix allow-list, re-enable |
| Cost overrun | Reduce `instance_count` or shorten TTL; never delete before removing affinity |

### 12.3 Cleanup after testing

```bash
kubectl delete ns rsv-test
kubectl delete k8sreservedpooltoleration reserved-pool-toleration
kubectl delete constrainttemplate k8sreservedpooltoleration
# gke workspace: remove rsv-test-* node pools, apply
# prj workspace: remove t-* entries per P7-11 (or let TTL expire, then remove entries)
gcloud compute reservations list --project "$OWNER_PROJECT" --filter="name~^${PREFIX}t-"   # expect empty
gcloud compute instances list --project "$OWNER_PROJECT" --filter="name~^rsv-"              # expect empty
gcloud projects remove-iam-policy-binding "$OWNER_PROJECT" \
  --member="serviceAccount:service-${CPNUM}@container-engine-robot.iam.gserviceaccount.com" \
  --role="projects/${OWNER_PROJECT}/roles/reservationReader"   # if not kept via Terraform
```

### 12.4 Evidence log template

Keep one row per test ID; attach files from `evidence/` to the change ticket.

| Test ID | Date | Executor | Result (Pass / Fail / Deviation) | Evidence file or TFC run URL | Notes / error text |
| --- | --- | --- | --- | --- | --- |
| P0-xx |  |  |  |  |  |
| P1-xx |  |  |  |  |  |
| P2-xx |  |  |  |  |  |
| P3-xx |  |  |  |  |  |
| P4-xx |  |  |  |  |  |
| P5-xx |  |  |  |  |  |
| P6-xx |  |  |  |  |  |
| P7-xx |  |  |  |  |  |
| P8-xx |  |  |  |  |  |
| P9-xx |  |  |  |  |  |

### 12.5 Open questions to close during testing

- Can one regional node pool consume three zone-specific reservation names, or do we need identical names per zone (P2-00)?
- Exact IAM permission the consumer GKE service agent needs for shared reservations (P5-03).
- Is G4 available in europe-west2, and which zones?
- A4 capacity model agreed with Google (future reservation, calendar mode or DWS)?
- Native Cloud Monitoring metric for reservation utilisation, or keep the exporter?

### 12.6 Sign-off

- [ ] Platform lead
- [ ] FinOps
- [ ] Security / IAM
- [ ] Pilot tenant representative
- [ ] CAB (prd change approved with this evidence pack)
