# Reservation Sizer – Build and Deploy Procedure (STE100)

Nov 5, 2026 · @Train17

This procedure installs Reservation Sizer kit v1.2 (`poc-1.2.0`) in one GKE cluster. It starts with the image build and ends with live mode.

## About this procedure

This procedure is for the platform engineer who installs the sizer. Do the procedure for one cluster at a time.

Do the tasks in this sequence:

| Task | Title | When to do it |
| --- | --- | --- |
| 1 | Prepare the Artifact Registry repository | One time for each repository |
| 2 | Build and test the image | For each new kit version |
| 3 | Push the image and record the digest | For each new kit version |
| 4 | Sign the image for Binary Authorization | For each new image, if your clusters enforce Binary Authorization |
| 5 | Prepare the GitHub App, the key secret and the CI guard | One time for each infra repository |
| 6 | Render the Kubernetes manifests | For each cluster, and after each change to the values |
| 7 | Deploy with Config Sync and examine the objects | For each cluster |
| 8 | Run the job one time and examine the result | After each deployment |
| 9 | Set the alerts | One time for each project |
| 10 | Change to dry-run mode | After the result of Task 8 is correct |
| 11 | Change to live mode | After 1 to 2 weeks of correct dry-run PRs |
| 12 | Install a new image version or roll back | When necessary |
| 13 | Stop the sizer (break-glass) | When necessary |

**Conventions (ASD-STE100):**

- Each step gives one instruction.
- A **CAUTION** tells you about a risk of damage to systems, data or cost. Read it before you do the step.
- A **NOTE** gives more information. It does not give an instruction.
- Text in `this font` is a command, a file, a field or a value. Type it exactly as shown.
- Text in `<angle brackets>` is a value that you must supply.
- The commands use the values in `config/sizer.env`. Load these values into your shell before each task:

```bash
cd reservation-sizer-poc
set -a; source config/sizer.env; set +a
```

## Prerequisites

Complete these kit steps (README v1.2) before you start Task 1:

- [ ] Step 0: the preflight shows `0 blocker(s)`.
- [ ] Step 1: the reservation map is in `node_pool_reservations.auto.tfvars.json`. The plan showed `No changes`, and the PR is merged.
- [ ] Step 2: the local report run is within the Step 2 gate.
- [ ] Step 3 and Step 3b: the service account, the custom role, the Workload Identity binding and the log view exist. The impersonated run is correct.

Install these tools on your workstation or CI runner:

| Tool | Used in |
| --- | --- |
| `gcloud` | All tasks |
| `docker` with `buildx` | Tasks 2 and 3 |
| `python3` 3.9 or later, `git`, `bash` | Tasks 2, 5 and 6 |
| `kubectl`, with a context for the cluster | Tasks 7, 8 and 10 to 13 |
| `nomos` (optional) | Task 7, to show the Config Sync status |
| `gh` (optional) | Task 5 |

Make sure that you have this access:

| Access | Used in |
| --- | --- |
| Artifact Registry administrator on the registry project (or a repository that you have) | Task 1 |
| `roles/artifactregistry.writer` on the repository | Task 3 |
| The roles that your organization uses to make attestations | Task 4 |
| GitHub organization owner (or GitHub App manager), and administrator of the infra repository | Task 5 |
| `roles/secretmanager.admin` on the cluster project | Task 5 |
| Write access to the Config Sync repository | Tasks 7 and 10 to 13 |
| `kubectl get` in the namespace, and `create` on Jobs | Task 8 |
| `roles/logging.configWriter` and `roles/monitoring.alertPolicyEditor`, or your Terraform pipeline | Task 9 |

Add these values to the end of `config/sizer.env`. The kit scripts do not read them. Only the commands in this document use them.

```bash
# ---- this procedure (build, push, deploy) ----------------------------------------
GAR_PROJECT="<registry-project>"        # project that holds the Artifact Registry repository
GAR_LOCATION="<region>"                 # for example europe-west2
GAR_REPO="<repository>"                 # for example platform-images
IMAGE_TAG="poc-1.2.0"
BASE_IMAGE="<approved-registry>/python:3.12-slim"
PIP_INDEX_URL=""                         # internal PyPI mirror; empty = public PyPI
CONFIG_REPO_DIR="<path to your Config Sync repository checkout>"
CONFIG_REPO_SUBDIR="<directory for this cluster's objects>"
```

Find the service account of the system node pool. The nodes use it to pull the image.

```bash
gcloud container node-pools describe "$SYSTEM_POOL" --cluster "$CLUSTER_NAME" \
  --region "$CLUSTER_LOCATION" --project "$PROJECT_ID" --format='value(config.serviceAccount)'
```

**NOTE:** If the result is `default`, the nodes use the Compute Engine default service account, `<project-number>-compute@developer.gserviceaccount.com`.

Write the result in `config/sizer.env`:

```bash
NODE_SA="<node service account email>"
```

## Task 1 – Prepare the Artifact Registry repository

This task makes a Docker repository and gives the necessary read and write access. If your platform team has a repository for platform images, use it. Then do only steps 5 to 8.

1. Load the values:

   ```bash
   set -a; source config/sizer.env; set +a
   ```
2. Enable the Artifact Registry API in the registry project:

   ```bash
   gcloud services enable artifactregistry.googleapis.com --project "$GAR_PROJECT"
   ```
3. Find out if the repository exists:

   ```bash
   gcloud artifacts repositories describe "$GAR_REPO" \
     --location "$GAR_LOCATION" --project "$GAR_PROJECT"
   ```
4. If the command shows `NOT_FOUND`, make the repository:

   ```bash
   gcloud artifacts repositories create "$GAR_REPO" --project "$GAR_PROJECT" \
     --location "$GAR_LOCATION" --repository-format=docker --immutable-tags \
     --description="Platform images"
   ```

   **NOTE:** With `--immutable-tags`, a push cannot move a tag that is in use to a different image. If your policy requires a customer-managed key, add `--kms-key=<key resource name>`.
5. Give the node service account permission to pull images:

   ```bash
   gcloud artifacts repositories add-iam-policy-binding "$GAR_REPO" \
     --project "$GAR_PROJECT" --location "$GAR_LOCATION" \
     --member="serviceAccount:$NODE_SA" --role=roles/artifactregistry.reader
   ```
6. Give the identity that pushes the image (your account or the CI service account) permission to write:

   ```bash
   gcloud artifacts repositories add-iam-policy-binding "$GAR_REPO" \
     --project "$GAR_PROJECT" --location "$GAR_LOCATION" \
     --member="<user:you@example.com or serviceAccount:ci@project.iam.gserviceaccount.com>" \
     --role=roles/artifactregistry.writer
   ```
7. If your organization uses vulnerability scanning, enable it in the registry project:

   ```bash
   gcloud services enable containerscanning.googleapis.com --project "$GAR_PROJECT"
   ```
8. Make sure that the cluster nodes can reach Artifact Registry:
   - The node subnet has Private Google Access.
   - If you use the private or restricted VIP, DNS for `pkg.dev` and `*.pkg.dev` points to the same VIP as `googleapis.com`.
   - If the cluster project is in a VPC Service Controls perimeter, the registry project is in the same perimeter, or an ingress rule lets the nodes pull.

**Result:** The repository exists. The nodes can read from it, and you can write to it.

## Task 2 – Build and test the image

This task builds the image on your workstation or CI runner and tests it with the offline demo data. It does not push the image.

1. Extract `reservation-sizer-poc-kit-v1.2.zip` into an empty directory.
2. Go to the kit directory:

   ```bash
   cd reservation-sizer-poc
   ```
3. Make sure that the kit version is `poc-1.2.0`:

   ```bash
   grep '^VERSION' sizer/reservation_sizer.py
   ```
4. Install the test dependency in a virtual environment:

   ```bash
   python3 -m venv .venv && . .venv/bin/activate
   pip install "PyJWT[crypto]>=2.8,<3"
   ```
5. Run the two test suites:

   ```bash
   python3 -m unittest discover -s sizer/tests
   python3 -m unittest discover -s tests
   ```
6. Make sure that each suite shows `OK`. The first suite runs 47 tests. The second suite runs 39 tests.

   **CAUTION:** DO NOT BUILD THE IMAGE IF A TEST FAILS. A failed test can cause incorrect reservation counts.
7. Write the full image name in `config/sizer.env`. Use the tag for now. Task 3 changes it to the digest.

   ```bash
   IMAGE="<GAR_LOCATION>-docker.pkg.dev/<GAR_PROJECT>/<GAR_REPO>/reservation-sizer:poc-1.2.0"
   ```

   **CAUTION:** WRITE THE VALUE AS PLAIN TEXT. DO NOT USE `${...}` IN `IMAGE`. The render script (Task 6) does not expand variables.
8. Load the values again:

   ```bash
   set -a; source config/sizer.env; set +a
   ```
9. Let Docker use your `gcloud` credentials for Artifact Registry:

   ```bash
   gcloud auth configure-docker "${GAR_LOCATION}-docker.pkg.dev"
   ```
10. If your base image is in a different registry, log in to that registry.
11. Build the image for `linux/amd64` and load it into the local Docker engine:

    ```bash
    docker buildx build --platform linux/amd64 \
      --build-arg BASE_IMAGE="$BASE_IMAGE" \
      --build-arg PIP_INDEX_URL="$PIP_INDEX_URL" \
      -t "$IMAGE" -f sizer/Dockerfile sizer --load
    ```

    **NOTE:** The kit script `setup/04-build-push.sh` does not send `PIP_INDEX_URL`. Use this command if you use an internal PyPI mirror.
12. Run the image with the demo data. Use the same restrictions as the cluster: user 10001 and a read-only root file system.

    ```bash
    docker run --rm --platform linux/amd64 --user 10001:10001 --read-only --tmpfs /tmp \
      -v "$PWD/examples/demo:/demo:ro" "$IMAGE" \
      --fixtures /demo --policy /demo/policy.json --tfvars-file /demo/tfvars.json \
      --project demo --location test-region1 --cluster c1 --env-name demo \
      --now 2026-11-04T06:00:00Z --out-dir /tmp/out
    echo "exit=$?"
    ```
13. Make sure that the output shows `exit=0` and two rows with these values (the spaces can be different):

    ```text
    k-a  test-region1-a  m-std-8  14  14  16  16  INCREASE
    k-b  test-region1-b  m-std-8  13  13  16  16  INCREASE
    ```
14. Make sure that the image runs as user 10001:

    ```bash
    docker image inspect "$IMAGE" --format '{{.Config.User}}'
    ```

    The result must be `10001:10001`.

**Result:** You have a tested image in the local Docker engine.

## Task 3 – Push the image and record the digest

This task pushes the tested image to Artifact Registry. Then it changes `IMAGE` to the digest form, so that the cluster pulls exactly the image that you tested.

1. Load the values:

   ```bash
   set -a; source config/sizer.env; set +a
   ```
2. Push the image:

   ```bash
   docker push "$IMAGE"
   ```

   **NOTE:** If the push shows that the tag is immutable, the tag is already in use. Do not delete the old image. Use a new tag, for example `poc-1.2.0-2`, and do Task 2 again from step 7.
3. Find the full image reference with the digest:

   ```bash
   gcloud artifacts docker images describe "$IMAGE" \
     --format='value(image_summary.fully_qualified_digest)'
   ```

   The result has this format: `<GAR_LOCATION>-docker.pkg.dev/<GAR_PROJECT>/<GAR_REPO>/reservation-sizer@sha256:<64 characters>`.
4. Replace the value of `IMAGE` in `config/sizer.env` with this result:

   ```bash
   IMAGE="<GAR_LOCATION>-docker.pkg.dev/<GAR_PROJECT>/<GAR_REPO>/reservation-sizer@sha256:<digest>"
   ```

   **CAUTION:** DO NOT DEPLOY THE IMAGE BY TAG. If the tag points to a different image later, the cluster runs an image that you did not test. Binary Authorization also requires the digest.
5. Load the values again:

   ```bash
   set -a; source config/sizer.env; set +a
   ```
6. If vulnerability scanning is enabled, wait 5 minutes. Then show the scan result:

   ```bash
   gcloud artifacts docker images describe "$IMAGE" --show-package-vulnerability
   ```
7. If the result shows CRITICAL or HIGH vulnerabilities, follow your organization's image policy before you continue.
8. Record these items in the change record:
   - the tag and the digest;
   - the result of the two test suites (Task 2, step 6);
   - the scan result.

**Result:** The image is in Artifact Registry, and `config/sizer.env` refers to it by digest.

## Task 4 – Sign the image for Binary Authorization

This task makes an attestation for the image digest. Do this task only if the cluster enforces Binary Authorization.

1. Show the Binary Authorization mode of the cluster:

   ```bash
   gcloud container clusters describe "$CLUSTER_NAME" --region "$CLUSTER_LOCATION" \
     --project "$PROJECT_ID" --format='value(binaryAuthorization.evaluationMode)'
   ```
2. If the result is empty or `DISABLED`, go to Task 5.
3. If your CI pipeline signs images, do not sign by hand. Go to step 6 and make sure that the attestation exists.
4. Get these values from your security team, and set them in your shell:

   ```bash
   ATTESTOR="<attestor name>"
   ATTESTOR_PROJECT="<project of the attestor>"
   KMS_PROJECT="<project of the signing key>"
   KMS_LOCATION="<key location>"
   KMS_KEYRING="<key ring>"
   KMS_KEY="<key>"
   KMS_KEY_VERSION="<key version>"
   ```
5. Sign the image digest and make the attestation:

   ```bash
   gcloud container binauthz attestations sign-and-create \
     --project="$ATTESTOR_PROJECT" --artifact-url="$IMAGE" \
     --attestor="$ATTESTOR" --attestor-project="$ATTESTOR_PROJECT" \
     --keyversion-project="$KMS_PROJECT" --keyversion-location="$KMS_LOCATION" \
     --keyversion-keyring="$KMS_KEYRING" --keyversion-key="$KMS_KEY" \
     --keyversion="$KMS_KEY_VERSION"
   ```

   **CAUTION:** SIGN ONLY THE DIGEST THAT YOU TESTED IN TASK 2 AND RECORDED IN TASK 3. An attestation lets the cluster run that image.
6. Show the attestations for the image:

   ```bash
   gcloud container binauthz attestations list --attestor="$ATTESTOR" \
     --attestor-project="$ATTESTOR_PROJECT" --artifact-url="$IMAGE"
   ```
7. Make sure that the list shows one attestation or more.

**NOTE:** If the policy is in dry-run mode, a denial only writes an audit log entry. Task 8 tells you how to find denials.

**Result:** The cluster can admit the image.

## Task 5 – Prepare the GitHub App, the key secret and the CI guard

This task makes a GitHub App for the sizer only, keeps its private key in Secret Manager and installs the CI guard. The job uses the App in all modes, because it reads the reservations file from GitHub.

**Make the GitHub App:**

1. In GitHub, go to your organization **Settings > Developer settings > GitHub Apps**.
2. Click **New GitHub App**.
3. Type a name, for example `reservation-sizer-<org>`. Type the infra repository URL as the homepage URL.
4. Under **Webhook**, clear **Active**.
5. Under **Repository permissions**, set these permissions only:
   - **Contents:** Read and write
   - **Pull requests:** Read and write
   - **Metadata:** Read-only

   **CAUTION:** DO NOT GIVE THE APP THE WORKFLOWS PERMISSION. With that permission, a stolen token can change the guard workflow and remove the guard.
6. Under **Where can this GitHub App be installed?**, select **Only on this account**.
7. Click **Create GitHub App**. Record the **App ID**.
8. Click **Generate a private key**. Save the file as `app-key.pem`.
9. Click **Install App**. Select **Only select repositories**, and select the infra repository only.
10. Record the installation ID. It is the number at the end of the URL `.../settings/installations/<ID>`.

**Keep the key in Secret Manager:**

11. Set these values in `config/sizer.env`:

    ```bash
    GITHUB_REPO="<owner>/<infra-repo>"
    GITHUB_BASE_BRANCH=""            # empty = default branch
    TFVARS_PATH="<workspace-dir>/node_pool_reservations.auto.tfvars.json"
    GITHUB_APP_ID="<App ID>"
    GITHUB_APP_INSTALLATION_ID="<installation ID>"
    GITHUB_APP_SECRET="reservation-sizer-github-key"
    ```
12. Load the values:

    ```bash
    set -a; source config/sizer.env; set +a
    ```
13. Put the key in Secret Manager, in the cluster region only:

    ```bash
    gcloud secrets create "$GITHUB_APP_SECRET" --project "$PROJECT_ID" \
      --replication-policy=user-managed --locations="$CLUSTER_LOCATION" --data-file=app-key.pem
    ```
14. Delete the local key file:

    ```bash
    shred -u app-key.pem 2>/dev/null || rm -f app-key.pem
    ```

    **CAUTION:** DO NOT KEEP A COPY OF THE KEY ON A DISK OR IN A CHAT. If the key is lost, generate a new key in the App settings.
15. Show the access commands. Copy them into the change record:

    ```bash
    DRY_RUN=1 ./setup/03-access-gcloud.sh
    ```
16. Run the access script again. It gives the sizer service account read access to this secret only:

    ```bash
    ./setup/03-access-gcloud.sh
    ```

**Install the CI guard in the infra repository:**

17. Copy the guard files into a branch of the infra repository:

    ```bash
    cp ci/check_reservation_change.py <infra-repo>/ci/
    cp ci/reservation-sizer-guard.yml <infra-repo>/.github/workflows/
    ```
18. Add this line to `<infra-repo>/.github/CODEOWNERS`:

    ```text
    /<TFVARS_PATH> @<org>/<platform-team>
    ```
19. Merge these changes with a normal PR.
20. Set the repository variables:

    ```bash
    gh variable set SIZER_TFVARS_PATH --repo "$GITHUB_REPO" --body "$TFVARS_PATH"
    gh variable set SIZER_MAX_STEP --repo "$GITHUB_REPO" --body 3
    ```

    **NOTE:** `SIZER_MAX_STEP` must be the same as `maxStepPerKeyPerPR` in `config/policy.json`.
21. In the branch protection rule (or ruleset) of the default branch, set these items:
    - Require the status check `reservation-sizer-guard`.
    - Require 1 approval and a review from code owners.
    - Dismiss stale pull request approvals when new commits are pushed.
    - Do not put the App in the bypass list.
22. Test the guard. Open a PR that changes one line in a README file, with the title `[DRY RUN] guard test`.
23. Make sure that `reservation-sizer-guard` fails on this PR. Then close the PR.

**Result:** The App can open PRs in the infra repository only, the key is in Secret Manager, and the guard protects the default branch.

## Task 6 – Render the Kubernetes manifests

This task makes the manifests for one cluster from `config/sizer.env` and `config/policy.json`.

1. If `config/policy.json` does not exist, copy it from the example:

   ```bash
   cp config/policy.example.json config/policy.json
   ```
2. In `config/policy.json`, set `"pools"` to the names of the in-scope node pools.
3. Make sure that `"mode"` is `"report"`.

   **CAUTION:** USE REPORT MODE FOR THE FIRST DEPLOYMENT. Dry-run and live modes write to GitHub.
4. In `config/sizer.env`, set the network values:

   | Value | Set it to |
   | --- | --- |
   | `GOOGLE_APIS_CIDR` | `199.36.153.8/30` for the private VIP, or `199.36.153.4/30` for the restricted VIP. Use the VIP that your DNS uses. |
   | `HTTPS_PROXY_URL` | The egress proxy URL for GitHub, for example `http://<proxy-host>:3128`. Empty = no proxy. |
   | `EGRESS_PROXY_CIDR` and `EGRESS_PROXY_PORT` | The proxy IP address as `/32`, and its port |
   | `NO_PROXY_LIST` | Keep the default. Google APIs must not go through the proxy. |
   | `PRIORITY_CLASS` | A PriorityClass for platform jobs that is in the cluster, or empty |
   | `SCHEDULE` | A daily time in UTC, off the hour. The default is `17 6 * * *`. |
5. Render the manifests:

   ```bash
   python3 setup/05-render-k8s.py
   ```
6. Make sure that the command wrote 4 files in `k8s/rendered/`: `serviceaccount.yaml`, `cronjob.yaml`, `networkpolicy.yaml` and `configmap-policy.yaml`.
7. Make sure that the CronJob uses the image digest:

   ```bash
   grep 'image:' k8s/rendered/cronjob.yaml
   ```

   The value must contain `@sha256:`.
8. If your Config Sync repository does not declare the namespace, write `k8s/rendered/namespace.yaml`:

   ```yaml
   apiVersion: v1
   kind: Namespace
   metadata:
     name: platform-capacity          # = NAMESPACE in config/sizer.env
     labels:
       pod-security.kubernetes.io/enforce: restricted
   ```

   **NOTE:** The sizer pod meets the `restricted` Pod Security Standard.
9. If the egress proxy inspects TLS, add your corporate root CA. Do steps 10 to 12. If not, go to step 13.
10. Make a ConfigMap manifest from the CA file:

    ```bash
    kubectl create configmap sizer-ca -n "$NAMESPACE" --from-file=ca.pem=<corporate-root-ca.pem> \
      --dry-run=client -o yaml > k8s/rendered/configmap-ca.yaml
    ```
11. In `k8s/rendered/cronjob.yaml`, add these lines to the container:

    ```yaml
                    env:
                      - { name: CA_BUNDLE, value: /etc/sizer-ca/ca.pem }   # add to the existing env list
                    volumeMounts:
                      - { name: corp-ca, mountPath: /etc/sizer-ca, readOnly: true }   # add to the existing list
    ```
12. In the same file, add this volume to the `volumes` list:

    ```yaml
                  - name: corp-ca
                    configMap:
                      name: sizer-ca
    ```

    **CAUTION:** THE RENDER SCRIPT OVERWRITES `cronjob.yaml`. After each render, do steps 11 and 12 again.

    **NOTE:** Kit v1.2 adds the CA file to the public root certificates. Direct calls to Google APIs continue to work.
13. Do a client-side test of the manifests:

    ```bash
    kubectl apply --dry-run=client -f k8s/rendered/
    ```
14. Make sure that the command shows no error.

**Result:** `k8s/rendered/` holds the manifests for this cluster, in report mode.

## Task 7 – Deploy with Config Sync and examine the objects

This task puts the manifests in the Config Sync repository. Config Sync then applies them to the cluster.

**CAUTION:** DO NOT APPLY THE MANIFESTS WITH `kubectl apply`. Config Sync reverts changes that are not in Git, and Git keeps no record of them.

1. If Policy Controller limits the image repositories, make sure that the constraint lets the cluster use `<GAR_LOCATION>-docker.pkg.dev/<GAR_PROJECT>/<GAR_REPO>`.
2. Make sure that the cluster enforces NetworkPolicy (Dataplane V2, or network policy enforcement on). If not, the NetworkPolicy has no effect.
3. In the Config Sync repository, make a new branch:

   ```bash
   cd "$CONFIG_REPO_DIR"
   git checkout -b "reservation-sizer-${CLUSTER_NAME}-poc-1.2.0"
   ```
4. Copy the rendered manifests into the directory for this cluster:

   ```bash
   mkdir -p "$CONFIG_REPO_SUBDIR"
   cp <kit-dir>/k8s/rendered/*.yaml "$CONFIG_REPO_SUBDIR/"
   ```
5. If the repository serves more than one cluster, limit the objects to this cluster. Use the method that your repository uses. For example, in an unstructured repository, add this annotation to each object:

   ```yaml
   metadata:
     annotations:
       configsync.gke.io/cluster-name-selector: <CLUSTER_NAME>
   ```
6. Commit and push the branch:

   ```bash
   git add "$CONFIG_REPO_SUBDIR"
   git commit -m "reservation-sizer poc-1.2.0 for ${CLUSTER_NAME} (report mode)"
   git push -u origin HEAD
   ```
7. Open a PR in the Config Sync repository. Merge it after the review.
8. Wait until Config Sync shows the new commit as synced:

   ```bash
   nomos status --contexts "$(kubectl config current-context)"
   ```

   **NOTE:** If you do not have `nomos`, use the Config Management page of the Google Cloud console.
9. Show the objects:

   ```bash
   kubectl -n "$NAMESPACE" get serviceaccount,configmap,cronjob,networkpolicy
   ```
10. Make sure that these objects exist:
    - the ServiceAccount `reservation-sizer`;
    - the ConfigMap `sizer-policy` (and `sizer-ca`, if you did Task 6 step 10);
    - the CronJob `reservation-sizer`;
    - the NetworkPolicy `reservation-sizer`.
11. Make sure that the ServiceAccount points to the sizer service account:

    ```bash
    kubectl -n "$NAMESPACE" get sa "$KSA_NAME" \
      -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}'; echo
    ```

    The result must be `<GSA_NAME>@<PROJECT_ID>.iam.gserviceaccount.com`.
12. Make sure that the CronJob uses the digest from Task 3:

    ```bash
    kubectl -n "$NAMESPACE" get cronjob reservation-sizer \
      -o jsonpath='{.spec.jobTemplate.spec.template.spec.containers[0].image}'; echo
    ```
13. Make sure that the policy mode is `report`:

    ```bash
    kubectl -n "$NAMESPACE" get configmap sizer-policy -o jsonpath='{.data.policy\.json}' \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["mode"])'
    ```
14. Show the schedule and the suspend state:

    ```bash
    kubectl -n "$NAMESPACE" get cronjob reservation-sizer -o jsonpath='{.spec.schedule} suspend={.spec.suspend}'; echo
    ```
15. Make sure that the schedule is correct and that `suspend` is not `true`.

**Result:** The CronJob is in the cluster, in report mode, with the tested image.

## Task 8 – Run the job one time and examine the result

This task starts one Job from the CronJob. It makes sure that the job uses Workload Identity and gives the same numbers as your laptop run in Step 3b.

1. Start one Job and show its log:

   ```bash
   ./setup/06-run-once-in-cluster.sh
   ```
2. If the pod does not start in 3 minutes, show the namespace events. Then refer to Troubleshooting.

   ```bash
   kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp | tail -20
   ```
3. When the log stops, find the name of the newest Job:

   ```bash
   JOB="$(kubectl -n "$NAMESPACE" get jobs --sort-by=.metadata.creationTimestamp -o name | tail -1)"
   echo "$JOB"
   ```
4. Show the exit code of the container:

   ```bash
   kubectl -n "$NAMESPACE" get pods -l job-name="${JOB#job.batch/}" \
     -o jsonpath='{.items[0].status.containerStatuses[0].state.terminated.exitCode}'; echo
   ```
5. Make sure that the exit code is `0`. If it is `3`, read the `hold:` lines in the log. A hold is correct while an apply is pending. For other codes, refer to Troubleshooting.
6. Make sure that the job used Workload Identity:

   ```bash
   kubectl -n "$NAMESPACE" logs "$JOB" | grep '"sizer run complete"' | grep -o '"token_source": "[a-z]*"'
   ```

   The result must be `"token_source": "metadata"`.
7. Show the in-cluster numbers:

   ```bash
   kubectl -n "$NAMESPACE" logs "$JOB" | grep '"sizer result"' | python3 -c '
   import json, sys
   for r in json.loads(sys.stdin.readline())["rows"]:
       print(r["key"], r.get("target_excl_stockout"), r.get("stockout_add"), r["status"])'
   ```
8. Show the numbers of your laptop run from Step 3b:

   ```bash
   python3 -c '
   import json
   for r in json.load(open("sizer-out/report.json"))["rows"]:
       print(r["key"], r.get("target_excl_stockout"), r.get("stockout_add"), r["status"])'
   ```
9. Compare the two lists. `target_excl_stockout` must be the same for each key.

   **NOTE:** `stockout_add` can be lower in the cluster. The new log bucket holds only the events after you made the sink in Step 3.
10. In the Cloud Logging Logs Explorer, run this query. Make sure that it finds the `sizer result` line:

    ```text
    resource.type="k8s_container"
    resource.labels.namespace_name="platform-capacity"
    jsonPayload.component="reservation-sizer"
    jsonPayload.message="sizer result"
    ```
11. If Binary Authorization is in dry-run mode, run this query. Make sure that it finds no entry for the sizer image:

    ```text
    labels."imagepolicywebhook.image-policy.k8s.io/dry-run"="true"
    ```
12. Make sure that the job opened no PR in the infra repository. Report mode does not write to GitHub.
13. Record the exit code, the token source and the two lists in the change record.

**NOTE:** The Job from step 1 deletes itself 7 days after it ends (`ttlSecondsAfterFinished`).

**Result:** The job runs in the cluster with Workload Identity and gives the same numbers as the laptop run.

## Task 9 – Set the alerts

This task adds three log-based alert policies with Terraform. They send messages to the channels that you use now (email, Microsoft Teams relay, Backstage).

| Alert | Log query finds | Action |
| --- | --- | --- |
| Run failed | An `ERROR` line from the sizer (exit 2, 4 or 5) | Do the Troubleshooting procedure on the same day |
| Key held | A `hold:` or `HOLD:` line (exit 3) | If the alert comes on 2 days in sequence, examine the HCP Terraform run |
| Pod problem | A Kubernetes `Warning` event in the namespace (image pull, admission, Job failure) | Do the Troubleshooting procedure |

1. Show the notification channels of the project and record the IDs that you want to use:

   ```bash
   gcloud beta monitoring channels list --project "$PROJECT_ID" \
     --format='table(name,displayName,type)'
   ```
2. Add this file to the Terraform workspace that holds the alert policies of the cluster project. Name it `reservation_sizer_alerts.tf`:

   ```hcl
   variable "sizer_namespace" {
     type    = string
     default = "platform-capacity"
   }
   variable "sizer_notification_channels" {
     type        = list(string)
     description = "Existing channel IDs: projects/<p>/notificationChannels/<id>"
   }
   
   locals {
     sizer_logs = <<-EOT
       resource.type="k8s_container"
       resource.labels.namespace_name="${var.sizer_namespace}"
       jsonPayload.component="reservation-sizer"
     EOT
     sizer_alerts = {
       run_failed = {
         name   = "Reservation sizer - run failed (exit 2, 4 or 5)"
         filter = "${local.sizer_logs}severity>=ERROR"
         doc    = "Read the ERROR lines. Do the Troubleshooting procedure."
       }
       key_held = {
         name   = "Reservation sizer - key held (exit 3)"
         filter = "${local.sizer_logs}jsonPayload.message=~\"^(hold: |HOLD: )\""
         doc    = "If this alert comes on 2 days in sequence, examine the HCP Terraform run."
       }
       pod_problem = {
         name   = "Reservation sizer - pod or Job warning"
         filter = <<-EOT
           logName="projects/${var.project_id}/logs/events"
           jsonPayload.involvedObject.namespace="${var.sizer_namespace}"
           jsonPayload.type="Warning"
         EOT
         doc    = "Read the event reason. Do the Troubleshooting procedure."
       }
     }
   }
   
   resource "google_monitoring_alert_policy" "sizer" {
     for_each     = local.sizer_alerts
     project      = var.project_id
     display_name = each.value.name
     combiner     = "OR"
     conditions {
       display_name = each.value.name
       condition_matched_log {
         filter = each.value.filter
       }
     }
     alert_strategy {
       notification_rate_limit {
         period = "3600s"
       }
       auto_close = "86400s"
     }
     notification_channels = var.sizer_notification_channels
     documentation {
       mime_type = "text/markdown"
       content   = each.value.doc
     }
   }
   ```

   **NOTE:** The file uses `var.project_id`. If your workspace uses a different variable name, change it.
3. Set `sizer_notification_channels` in the workspace variables.
4. Open a PR. Make sure that the speculative plan shows `3 to add, 0 to change, 0 to destroy`.
5. Merge the PR. Make sure that HCP Terraform applies it.
6. Test the pod-problem alert. Start a Job with an image digest that does not exist:

   ```bash
   kubectl -n "$NAMESPACE" create job sizer-alert-test \
     --image="${IMAGE%@*}@sha256:0000000000000000000000000000000000000000000000000000000000000000"
   ```
7. Make sure that the alert message arrives in your channels in 15 minutes or less.
8. Delete the test Job:

   ```bash
   kubectl -n "$NAMESPACE" delete job sizer-alert-test
   ```
9. In the Logs Explorer, run the queries of the other two alerts. Make sure that they show no syntax error.

**NOTE:** If you made the metric `reservation_sizer_errors` in Step 3 (`CREATE_LOG_METRIC=1`), you can keep it. These alerts do not use it.

**Result:** You get a message when a run fails, a key stays held, or the pod cannot run.

## Task 10 – Change to dry-run mode

In dry-run mode, the job opens a dummy PR with real numbers. The guard fails the PR on purpose, so nobody can merge it. Keep this mode for 1 to 2 weeks.

**CAUTION:** DO NOT MERGE A DRY-RUN PR. DO NOT REMOVE THE `[DRY RUN]` TITLE PREFIX OR THE `do-not-merge` LABEL. If you merge it, HCP Terraform increases the reservations and you pay for them.

1. Make sure that Task 8 gave a correct result and that the guard from Task 5 is installed.
2. In `config/policy.json`, set `"mode": "dry-run"`.
3. Render the manifests again:

   ```bash
   python3 setup/05-render-k8s.py
   ```
4. If you added the corporate CA in Task 6, do Task 6 steps 11 and 12 again.
5. Copy `k8s/rendered/configmap-policy.yaml` to the Config Sync repository directory for this cluster.
6. Show the change in the Config Sync repository:

   ```bash
   git -C "$CONFIG_REPO_DIR" diff
   ```
7. Make sure that only the line that you changed in `config/policy.json` is different.
8. Commit with the message `reservation-sizer: dry-run mode for <CLUSTER_NAME>`. Open a PR, and merge it after the review.
9. Wait until Config Sync shows the new commit as synced (Task 7, step 8).
10. Make sure that the policy mode is `dry-run` (Task 7, step 13).
11. Run the job one time (Task 8, steps 1 to 5).
12. If the job proposes an increase, show the PR:

    ```bash
    gh pr list --repo "$GITHUB_REPO" --head "reservation-sizer/${ENV_NAME}-${CLUSTER_NAME}"
    ```
13. Make sure that the PR has these properties:
    - The title starts with `[DRY RUN]`.
    - The PR is a draft, or it has the label `do-not-merge`.
    - The check `reservation-sizer-guard` fails.
    - The HCP Terraform speculative plan shows only in-place updates of reservation counts.
14. Compare the proposed counts with your manual numbers. Use the Step 2 method of the README.
15. On each work day, do steps 13 and 14 again for the updated PR. Record the result.

**NOTE:** The job updates the same PR each day. It pushes only when the counts change. When no increase is necessary, it closes the PR with a comment.

**Result:** After 1 to 2 weeks, you have a record of the proposed counts against your manual numbers.

## Task 11 – Change to live mode

In live mode, the job opens normal PRs. A person approves and merges each PR, and HCP Terraform applies it.

1. Make sure that the dry-run record (Task 10) shows 1 to 2 weeks of correct counts.
2. Get approval for the change, as your change process tells you.
3. In `config/policy.json`, set `"mode": "live"`.
4. Do Task 10, steps 3 to 9, with the commit message `reservation-sizer: live mode for <CLUSTER_NAME>`.
5. Make sure that the policy mode is `live` (Task 7, step 13).
6. Run the job one time (Task 8, steps 1 to 5), or wait for the schedule.
7. If the job proposes an increase, make sure of these items:
   - The dry-run PR is closed, with the comment "mode changed to live".
   - A new PR is open. Its title does not start with `[DRY RUN]`. It is not a draft, and it has no `do-not-merge` label.
   - The check `reservation-sizer-guard` passes.
   - The speculative plan shows only in-place updates of reservation counts. Each count increases by 3 or less.
8. Read the evidence table in the PR description.

   **CAUTION:** READ THE PLAN BEFORE YOU APPROVE. A merge starts an apply, and reserved VMs cost money from that time.
9. Approve and merge the PR, as CODEOWNERS requires.
10. Make sure that HCP Terraform applies the run. If your workspace has manual confirmation, confirm the apply.
11. If the apply fails with `ZONE_RESOURCE_POOL_EXHAUSTED` or another capacity error, do these steps:
    1. Wait for the next run. It holds the key (`PENDING_APPLY`, exit 3) and shows `step_back_to`.
    2. Open a human PR that sets `instance_count` of that key to the `step_back_to` value.
    3. Merge it after the review. Make sure that the apply succeeds.

    **NOTE:** The guard does not let bot PRs decrease a count. A person must make this step-back PR.
12. After the next scheduled run, show the result (Task 8, step 7). Make sure that each changed key shows `OK`, not `PENDING_APPLY`.
13. Do the items in **POC success criteria** in the README. Record the results.

**Result:** The sizer is live. Each day it proposes the increases that are necessary, and a person approves each one.

## Task 12 – Install a new image version or roll back

Use this task when a new kit version is available, or when a new version causes a problem.

**CAUTION:** DO NOT DELETE THE PREVIOUS IMAGE FROM ARTIFACT REGISTRY FOR 30 DAYS. It is necessary for a roll back. Make sure that no cleanup policy deletes it.

**Install a new version:**

1. Read the **What changed** section of the new README.
2. If the new version adds, removes or renames policy keys, change `config/policy.json` to agree. The sizer stops with exit 4 if the policy has an unknown key.
3. Set `IMAGE_TAG` in `config/sizer.env` to the new version.
4. Do Task 2 with the new kit.
5. Do Task 3. Record the new digest and the previous digest.
6. If the cluster enforces Binary Authorization, do Task 4.
7. Do Task 6.
8. Copy all files from `k8s/rendered/` to the Config Sync repository directory for this cluster.
9. Show the change:

   ```bash
   git -C "$CONFIG_REPO_DIR" diff
   ```
10. Make sure that the change contains the new image digest and only the changes that the README tells you about.
11. Commit with the message `reservation-sizer <version> for <CLUSTER_NAME>`. Open a PR, and merge it after the review.
12. Wait until Config Sync shows the new commit as synced (Task 7, step 8).
13. Make sure that the CronJob uses the new digest (Task 7, step 12).
14. Do Task 8.

**Roll back:**

15. In the Config Sync repository, find the commit that installed the new version:

    ```bash
    git -C "$CONFIG_REPO_DIR" log --oneline -- "$CONFIG_REPO_SUBDIR"
    ```
16. Revert that commit on a new branch:

    ```bash
    git -C "$CONFIG_REPO_DIR" checkout -b "rollback-reservation-sizer-${CLUSTER_NAME}" origin/main
    git -C "$CONFIG_REPO_DIR" revert <commit>
    git -C "$CONFIG_REPO_DIR" push -u origin HEAD
    ```
17. Open a PR, and merge it after the review.
18. Wait until Config Sync shows the revert commit as synced.
19. Make sure that the CronJob uses the previous digest (Task 7, step 12).
20. Do Task 8.

**NOTE:** A roll back does not change the reservations. It changes only the sizer image and its settings.

**Result:** The cluster runs the new version, or the previous version again.

## Task 13 – Stop the sizer (break-glass)

Use this task when the sizer proposes incorrect counts, or when you must stop all changes quickly. Do the steps in this sequence. The first step stops all writes to GitHub at once.

**CAUTION:** DO NOT USE `kubectl edit` OR `kubectl patch` ON THE CRONJOB OR THE CONFIGMAP. Config Sync reverts these changes in a few minutes.

**Stop:**

1. In GitHub, go to the organization **Settings > GitHub Apps** (installed apps).
2. Select the sizer App, click **Configure**, then click **Suspend**.
3. If a Job runs now, delete it. Jobs are not in Git, so Config Sync does not restore them.

   ```bash
   kubectl -n "$NAMESPACE" get jobs
   kubectl -n "$NAMESPACE" delete job <job-name>
   ```
4. Close the open bot PR, if there is one:

   ```bash
   gh pr close <number> --repo "$GITHUB_REPO" --comment "Sizer stopped: <reason>"
   ```
5. If a bot PR with incorrect counts is merged, do these steps:
   1. In HCP Terraform, discard the run if it is not applied.
   2. Revert the merge commit in the infra repository with a human PR.
   3. If the apply is complete, the revert decreases the reservation. Get approval before you merge it.

   **NOTE:** `prevent_destroy` stops any plan that deletes or replaces a reservation.
6. In `config/policy.json`, set `"enabled": false`.
7. Do Task 10, steps 3 to 9, with the commit message `reservation-sizer: stopped on <CLUSTER_NAME>`.

**NOTE:** With `"enabled": false`, each run stops at once and exits 0. The CronJob stays in the cluster.

**Start again:**

8. Find and correct the cause of the problem.
9. In `config/policy.json`, set `"enabled": true` and `"mode": "report"`.
10. Do Task 10, steps 3 to 9, with the commit message `reservation-sizer: restarted in report mode on <CLUSTER_NAME>`.
11. In GitHub, go to the App configuration again and click **Unsuspend**.
12. Do Task 8. Then do Task 10 and Task 11 again.

**Result:** The sizer makes no change until you start it again in report mode.

## Troubleshooting

Find the symptom in the table. Then do the action. Exit codes: `0` OK, `2` fail-safe, `3` hold, `4` configuration, `5` PR failed.

Show the error lines of the last Job:

```bash
kubectl -n "$NAMESPACE" logs "$(kubectl -n "$NAMESPACE" get jobs --sort-by=.metadata.creationTimestamp -o name | tail -1)" \
  | grep -E '"severity": "(ERROR|WARNING)"'
```

| Symptom | Possible cause | Action |
| --- | --- | --- |
| Pod shows `ImagePullBackOff` or `ErrImagePull` | The node service account cannot read the repository, the nodes cannot reach `pkg.dev`, or the digest is incorrect | Do Task 1, steps 5 and 8. Do Task 7, step 12. |
| Event `FailedCreate` with a Binary Authorization denial | The digest has no attestation | Do Task 4. |
| Event `FailedCreate` with `admission webhook ... denied` | A Policy Controller constraint stops the pod | Read the message. Change the manifest (Task 6), or get an exemption. |
| Event `FailedCreate` with `violates PodSecurity` | The namespace level is stricter than `restricted`, or somebody changed the manifest | Compare the CronJob in Git with `k8s/rendered/cronjob.yaml`. |
| Pod stays `Pending` with `FailedScheduling` | `SYSTEM_POOL` is incorrect, or the pool has no free capacity | Set `SYSTEM_POOL`. Do Task 6 and Task 7. |
| Exit 4: `no credentials: metadata server unreachable` | Workload Identity is off on the node pool, the ServiceAccount annotation is incorrect, or the NetworkPolicy stops the metadata server | Do Task 7, step 11. Examine the metadata-server rule in `networkpolicy.yaml`. |
| Exit 2: `HTTP 403` from Monitoring, GKE or Compute Engine | The custom role or its binding is missing | Run `./setup/03-access-gcloud.sh` again. |
| Exit 2: `HTTP 403` on `entries:list` | The conditional `viewAccessor` binding does not match the log view | Run `./setup/03-access-gcloud.sh` again. Make sure that the `LOG_*` values did not change. |
| Exit 2: `TLS verification failed` | The proxy inspects TLS and the job has no corporate CA | Do Task 6, steps 9 to 12. |
| Exit 2: `network error` to `googleapis.com` | `GOOGLE_APIS_CIDR` is not the VIP that DNS gives, or Google API calls go through the proxy | Do Task 6, step 4. Keep `.googleapis.com` in `NO_PROXY_LIST`. |
| Exit 2: `coverage ... (< 90%)` | The metric history has gaps, or the cluster is new | Run the preflight (README Step 0). Wait until 7 days of history exist. |
| Exit 2: `machine_type ... but reservation ... is` | The reservations file and the live reservation do not agree | Correct the file with a human PR. |
| Exit 3: `branch says X, Compute Engine has Y` | The apply is pending, or it failed | Do Task 11, step 11. |
| Exit 3: `changed outside Terraform` | Somebody changed the reservation outside Terraform (`DRIFT`) | Make a human PR that sets the file to the live count. |
| Exit 4: `unknown policy keys` or `policy.mode must be` | `config/policy.json` has an error | Correct the file. Do Task 10, steps 3 to 9. |
| Exit 4: `node pools in policy not found` | A pool was renamed or made again | Correct `"pools"`. Do Task 10, steps 3 to 9. |
| Exit 4 or 5: GitHub `HTTP 401` or `HTTP 404` | The App ID, installation ID or key is incorrect, or the App is suspended | Do Task 5, steps 10 to 16. |
| Exit 5: `PR raise failed` with a network error | The job cannot reach `api.github.com` through the proxy | Examine `HTTPS_PROXY_URL` and `EGRESS_PROXY_CIDR` (Task 6, step 4). |
| The guard fails on a live bot PR | The PR changes more than counts, or `SIZER_MAX_STEP` is not equal to `maxStepPerKeyPerPR` | Read the check log. Make the two values equal. |
| `nomos status` shows an error | The Config Sync repository has an error | Correct the repository. Wait for the sync. |
| No Job at the scheduled time | The CronJob is suspended, or the start time was missed by more than 1 hour | Do Task 7, step 14. |
