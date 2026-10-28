# GKE Capacity Report Tool Guide

Oct 28, 2026 · @S

This tool tells us how many nodes of each machine shape we really use in each zone, and turns that into a Capacity Reservation Table we can paste straight into Terraform. It is read-only, so it is safe to run against prd: it only describes, lists and reads, and never changes clusters, reservations or quotas. This guide covers version 2026-10-28.1.

## What the tool is for

A Compute Engine reservation holds one machine shape in one zone. So to size reservations properly, we need to know the highest number of nodes of each shape that ran in each zone, not just a cluster total. That is the one question this tool answers, and everything else in it supports that answer.

The tool has two files, and they must sit in the same folder (we keep them in `reservation_poc`):

| File | What it does |
| --- | --- |
| `gke_node_peaks.py` | The Python script. It reads GKE, Compute Engine, Cloud Monitoring and Cloud Logging, works out the peaks and writes CSV files, a JSON file and the HTML report. |
| `gke_capacity_report.html` | The report template. The script copies it and puts the data inside, so the final `report.html` opens in any browser with no server and no internet. You can also open the template on its own and load `report_data.json` files into it. |

The script needs only Python 3.8 or later and an authenticated `gcloud`. `kubectl` is optional. No extra Python packages are needed.

## What the script captures

The script runs in four steps, one after the other, for each cluster you give it.

1. **Live inventory (step 1).** It reads every node pool from GKE: machine type, min CPU platform, GPUs, Local SSD, Spot flag, reservation affinity, autoscaler min and max, location policy, maxSurge and maxUnavailable. Then it lists the node VMs from Compute Engine and counts them by pool, zone, machine type and status. It records the machine type the VM itself reports, because that is what a reservation must match. For example, a pool configured as `n2-custom-32-32768` can show up as `n2-highcpu-32`, and the script uses the second name. If you pass a kube context, it also checks the Kubernetes nodes against the VMs and lists NotReady and cordoned nodes.
2. **Capacity data (step 1b).** It reads the reservations that already exist in the region, how many slots each one has and how many are in use, plus the regional quotas and the vCPU and memory of each machine type. This tells us what we already pay for and whether a new plan fits in quota.
3. **Historical peaks (step 2).** It pulls one Cloud Monitoring series per node (`kubernetes.io/node/cpu/allocatable_cores`) and counts how many nodes existed in each 5-minute window. It then works out the peak, P95, mean and minimum per pool and zone, per shape and zone, per zone and for the whole cluster. The shape and zone peak is the highest number of nodes of that shape that ran in that zone at the same moment, across all pools, which is the right base for a reservation. The script also finds node upgrade windows and gives a second peak without them, and you can drop an outage period with `--exclude-window`.
4. **Autoscaler events (step 3).** It reads the cluster autoscaler visibility log and summarises scale-ups, scale-downs, failed scale-ups and stockouts (`scale.up.error.out.of.resources`) per pool and zone. A stockout here is real proof that the zone ran out of capacity. This step can be slow on busy clusters, so `--log-days 7` is a good start, and pressing Ctrl+C skips only this step because steps 1 and 2 are already saved.

At the end it builds the **Capacity Reservation Table**: for each shape and zone it calculates `ceil(peak x (1 + headroom %)) + surge`, compares that with what is already reserved, and gives a status such as ADD, OVER-RESERVED, IDLE or STOCKOUT. It also writes the same plan as a Terraform `node_pool_reservations` map.

## Output files

Each run creates a folder named `node-peaks-<cluster>-<timestamp>` (or `node-peaks-<project>-<timestamp>/<cluster>/` when you run several clusters together). These are the files you will find in it.

| File | Step | What it holds |
| --- | --- | --- |
| `capacity_reservation_table.csv` | Final | The plan: one row per shape and zone, with peak, calculation, recommended VMs, reserved now, in use, change and status |
| `node_pool_reservations.tfvars` | Final | The same plan as a Terraform map, in the same layout as our `prj/terraform.tfvars` |
| `report.html` | Final | The interactive report with the data built in |
| `report_data.json` | Final | All the data behind the report; use it to rebuild or merge reports later |
| `summary.json` | Final | Run settings, history range, node mapping quality and checks |
| `sizing_input_shape_zone.csv` | 2 | Peak, formula and sizing input per shape and zone, plus existing reservations |
| `peaks_by_pool_zone.csv`, `peaks_by_shape_zone.csv`, `peaks_by_pool.csv`, `peaks_by_zone.csv`, `peaks_by_shape.csv`, `peaks_by_cluster.csv` | 2 | Peak, peak time, peak without upgrades, P95, mean and min at each level |
| `timeseries_pool_zone.csv` | 2 | Node count per pool and zone for every 5-minute window, for your own charts |
| `node_mapping.csv` | 2 | How each node from Monitoring was matched to a pool and zone |
| `upgrade_operations.csv` | 2 | Node upgrade operations that were removed from the second peak |
| `autoscaler_events_pool_zone.csv` | 3 | Scale-ups, scale-downs, failed scale-ups, stockouts and no-scale-up reasons per pool and zone |
| `node_pools.csv` | 1 | Node pool settings, including affinity, autoscaler limits and surge |
| `live_vms.csv`, `live_counts_pool_zone.csv` | 1 | Every node VM now, and the counts per pool, zone and machine type |
| `k8s_nodes.csv` | 1 | Kubernetes nodes with Ready and cordon state (only with a kube context) |
| `existing_reservations.csv` | 1b | Reservations in the region with count, in use, idle and the pools each can serve |
| `regional_quotas.csv` | 1b | Quota limit, usage and what is left per metric |

## The HTML report

The report is one self-contained page. It uses no external scripts or fonts, and the data never leaves your browser, so it is fine to share inside the team. If you open the template without data, it shows sample numbers and a warning banner. These are the sections from top to bottom.

| Section | What it shows | How to use it |
| --- | --- | --- |
| Cluster selector | A dropdown of the clusters in the page, plus **All clusters combined** when there are two or more. A **Load report\_data.json** button adds more files. | Switch cluster here. The page remembers your last choice. |
| Cluster header | Project, history range, window size, node mapping accuracy, upgrade windows removed and excluded ranges | Check the history really covers the period you wanted before trusting any number. |
| Machine filter | Chips for CPU predefined, CPU custom and GPU, and a machine type dropdown | Every chart, table, total and download follows this filter. |
| KPI tiles | Running VMs, cluster peak, P95, recommended VMs, reserved now, change, idle reserved VMs, failed scale-ups and stockouts | A quick health check. Red tiles need attention. |
| Data flow | A picture of the sources, the three steps and the outputs | Explains the tool to someone new. |
| Current and peak usage | Bars per pool and zone for current, P95 and peak, with a marker for peak without upgrades | Spot zones that peak much higher than others. |
| **Capacity Reservation Table** | One row per reservation (shape and zone): status, pools, peak, calculation, recommended VMs, running now, reserved now, change, vCPU and notes, with subtotals and a grand total | The main output. Set headroom, base and surge at the top of this section, then download the CSV or the tfvars. |
| Existing reservations | Each reservation with reserved, in use, idle, utilisation, consumption type and the pools it can serve | Look for idle slots and reservations with no matching pool. |
| Quota check | Extra vCPU, GPU, instance and reservation quota the plan needs, against the regional limit | Raise quota before applying anything marked short. |
| Detail tables | Tabs for machine types to node pools, peaks by pool and zone, autoscaler activity and node pool settings | For digging into a specific pool or zone. |
| Metric guide | A plain explanation of every metric and how to use it | Share with anyone reviewing the numbers. |

The controls in the Capacity Reservation Table section are worth knowing:

- **Headroom:** 0 to 30 percent, or a custom value. Use 0 percent when you only want to protect what runs today.
- **Base:** peak, or peak without upgrade windows.
- **Upgrade surge:** largest maxSurge (our setting, because pools upgrade one at a time), sum of maxSurge (parallel upgrades) or none.
- **Reservation name prefix:** left empty, it is worked out from the project, for example `msv-pre-caas01-svp02-41428` gives `pre-svp02`. Type your own to override it.
- **Download CSV and Download tfvars:** both use exactly what you see on screen, including the filter.

The Status column reads like this:

| Status | Meaning | What to do |
| --- | --- | --- |
| OK | Reserved equals recommended | Nothing |
| ADD n | We need n more reserved VMs | Create or resize up |
| OVER-RESERVED n | We reserve n more than needed | Shrink, after checking upcoming growth |
| IDLE n | n reserved slots that no node uses right now | Normal at night for autoscaled pools; worrying if it stays high |
| NOT CONSUMED | Nodes are running, but the reservation is not used | Check the shape match first |
| STOCKOUT xn | The autoscaler hit out-of-resources n times in this zone | Highest priority to secure |
| NOT RESERVABLE | Spot or unmapped nodes | Spot VMs cannot use reservations |

## How to run it

**Before the first run**, log in to gcloud and, if you want the kubectl cross-check, fetch cluster credentials:

```bash
gcloud auth login
gcloud container clusters get-credentials <CLUSTER> --region europe-west2 --project <PROJECT>
```

**One cluster.** A target is written as `CLUSTER@LOCATION`. This example reads 30 days of peaks, 7 days of autoscaler logs, protects only today's footprint (0 percent headroom) and adds one surge VM per zone:

```bash
python3 gke_node_peaks.py --project <PROJECT> \
  --target <EW2_CLUSTER>@europe-west2 \
  --days 30 --log-days 7 --growth-pct 0 --surge-mode one --kube-context auto
```

**Both clusters in one project.** Our projects hold one cluster in EW2 and one in EW1, so repeat `--target`. The script runs them one by one, writes a subfolder for each, and also writes `report-all-clusters.html`. If one cluster fails, it carries on with the next.

```bash
python3 gke_node_peaks.py --project <PROJECT> \
  --target <EW2_CLUSTER>@europe-west2 \
  --target <EW1_CLUSTER>@europe-west1 \
  --days 30 --log-days 7 --growth-pct 0 --surge-mode one --kube-context auto
```

**Several projects.** One run covers one project, so run the script once per project (for example svp01 and svp02), then merge the results with `build-html`.

**Merge reports with build-html.** This does not call Google at all. It only combines existing `report_data.json` files into one page, so you can rerun it any time after updating the template:

```bash
python3 gke_node_peaks.py build-html --out pre-all-clusters.html \
  node-peaks-*/report_data.json node-peaks-*/*/report_data.json
```

You can also skip this command: open `gke_capacity_report.html` in a browser and use **Load report\_data.json** to pick several files.

**The combined view.** When a page holds two or more clusters, the dropdown shows **All clusters combined**. In this view:

- Pools are labelled by region, for example `ew1:user` and `ew2:systm`.
- The Capacity Reservation Table still has one row per shape and zone, with totals across both regions.
- Quota is checked separately for each project and region.
- The tfvars download writes one `node_pool_reservations` block per project, each with its own name prefix.
- If two clusters share the same project and region, a banner warns that their peaks were added and may overstate the joint peak.

**Getting the numbers into Terraform.** Download the tfvars (or take `node_pool_reservations.tfvars` from the output folder), compare it with `prj/terraform.tfvars`, and copy the entries across. Keys and names follow our existing pattern, for example key `pre-svp02-ew1-n4-c32-73728-d` with name `pre-svp02-ew1-n4-c32-73728`. Shapes with GPUs, Local SSD or a min CPU platform get a NOTE line, because our module only sets the machine type.

**Long runs: keep it running with tmux and caffeinate.** A full prd run (90 days, four clusters) can take a long time, and two things can stop it halfway: the terminal window closing, and the Mac going to sleep. tmux takes care of the first and caffeinate takes care of the second, so use both together.

```bash
# once
brew install tmux

# start a named session
tmux new -s prd-sizing

# inside tmux: keep the Mac awake for as long as the script runs, and save the output to a log
caffeinate -dims python3 gke_node_peaks.py --project <PRD_SVP01_PROJECT> \
  --target <EW2_CLUSTER>@europe-west2 --target <EW1_CLUSTER>@europe-west1 \
  --days 90 --log-days 30 --growth-pct 0 --surge-mode one --kube-context auto \
  2>&1 | tee prd-svp01-run.log
```

| Action | Command |
| --- | --- |
| Leave the run going and get your terminal back (detach) | Press Ctrl+B, then D |
| Come back to the run later | `tmux attach -t prd-sizing` |
| See which sessions are running | `tmux ls` |
| Watch progress from another window | `tail -f prd-svp01-run.log` |
| Close the session after the run finishes | `exit` inside tmux |

Please keep these points in mind:

- **tmux alone does not stop sleep.** It keeps the run alive if the terminal closes or you detach, but if the Mac sleeps, the script pauses and the connection to Google drops.
- **caffeinate stops idle sleep, not lid sleep.** With `-dims` the Mac stays awake until the script ends. Closing the lid on a MacBook still puts it to sleep unless it is plugged in with an external display, so keep the lid open and the charger connected.
- **A locked screen is fine; sleep is not.** Company policy may lock the screen after some minutes, and that does not affect the run.
- **For the most reliable prd run, use a Linux jump host or VM with tmux**, if the team has one with gcloud and the read-only roles. It never sleeps, and you can detach and go home.
- **If a run does stop**, simply start it again. The script only reads data, so a rerun is safe and writes a fresh output folder.

## Options reference

| Option | Default | What it does |
| --- | --- | --- |
| `--project` | required | The GCP project that holds the clusters |
| `--target CLUSTER@LOCATION` | none | A cluster and its region; repeat it for several clusters |
| `--cluster` with `--location` | none | The older way to name clusters; give one location per cluster, in the same order |
| `--days` | 30 | How many days of history to read for peaks |
| `--log-days` | same as `--days` | How many days of autoscaler logs to read; roughly 1 minute per 1 to 2 days on busy clusters |
| `--growth-pct` | 0 | Headroom percent added on top of the peak in the CSV and tfvars |
| `--basis` | `peak` | `peak`, or `peak_excl_upgrades` to ignore upgrade windows |
| `--surge-mode` | `one` | `one` adds the largest maxSurge per shape and zone, `all` adds the sum, `none` adds nothing |
| `--include-surge` | off | Old option, now the same as `--surge-mode all` |
| `--tfvars-prefix` | from project | Prefix for reservation names, for example `pre-svp02` |
| `--exclude-window START,END` | none | Removes a UTC time range from the history, for example an outage; repeatable |
| `--kube-context` | none | `auto` (the name gcloud get-credentials creates), `current`, or a context name, for the kubectl cross-check |
| `--alignment` | 300 | Seconds per counting window; minimum 60 |
| `--chunk-days` | 1 | Days per Monitoring API request; lower it if requests time out |
| `--page-size` | 100000 | Data points per Monitoring API page |
| `--log-limit` | 100000 | Maximum autoscaler log entries to read |
| `--log-noscaleup` | off | Also reads noScaleUp events; very high volume and much slower |
| `--skip-monitoring` | off | Skips step 2 (historical peaks) |
| `--skip-logs` | off | Skips step 3 (autoscaler logs) |
| `--skip-capacity` | off | Skips step 1b (reservations, quotas, machine specs) |
| `--ca-bundle` | none | PEM file with the company root certificate, for TLS-inspecting proxies |
| `--out-dir` | timestamped folder | Where to write the outputs |
| `--html-template` | file next to the script | A different report template |

The `build-html` command takes `--out` (default `gke-capacity-report.html`), `--template`, and then the list of `report_data.json` files.

## Permissions, network and common problems

The account running the script needs read-only roles on each project: `roles/container.viewer`, `roles/compute.viewer`, `roles/monitoring.viewer` and `roles/logging.viewer`. Nothing more is needed, because the script never writes to Google Cloud.

| Problem | Likely cause | Fix |
| --- | --- | --- |
| TLS certificate verification failed | The company proxy inspects HTTPS and Python does not trust its root certificate, even though gcloud works | Export the company root certificate as a PEM file and pass `--ca-bundle ~/corp-root.pem`, or set `SSL_CERT_FILE`. Never turn off verification. |
| Monitoring requests time out | A day of data per request is too big for the proxy | Use `--chunk-days 1` (the default) or a smaller `--page-size` |
| History starts later than requested | Metric retention, or a new cluster | Read the warning; peaks only cover the period shown in the header |
| Coverage check warning | Monitoring saw a different node count from Compute Engine in the last window | Investigate before trusting the peaks |
| Many heuristic or unmapped nodes | Nodes from pools that no longer exist | Check `node_mapping.csv`; a high count lowers accuracy |
| Step 3 is very slow | Busy clusters write a lot of autoscaler logs | Use `--log-days 7`, or press Ctrl+C to skip it; steps 1 and 2 are kept |
| Hit the log limit warning | More log entries than `--log-limit` | Use fewer `--log-days` or raise `--log-limit` |
| kubectl context does not match | The wrong context was used | Use `--kube-context auto`, or run get-credentials first |

The script reads gcloud's own proxy settings and the `HTTPS_PROXY` variable, so it normally works wherever gcloud works.

## Limits to keep in mind

- **A recommendation is a starting point, not an approval.** Review each row, especially OVER-RESERVED ones, before changing Terraform.
- **A reservation holds VM capacity only.** It does not schedule pods; pods still need correct requests, tolerations and quota.
- **The history window matters.** Thirty days may miss month-end, salary day or seasonal peaks. For prd sizing, use 90 days or a known busy period.
- **Specific affinity hides real demand.** A pool on a specific reservation can never grow past it, so its history is capped and real demand may be higher.
- **Zone peaks do not always happen together.** If a zone failed over or rebalanced, the sum of zone peaks can be higher than the shape ever ran at once. The report warns about this under the table, and `--exclude-window` can remove an outage.
- **Blue-green upgrades are not modelled.** They need about twice the nodes during the upgrade; the table notes it but does not add it.
- **The tfvars only sets machine type.** Shapes that also need GPUs, Local SSD or a min CPU platform are flagged with a NOTE and need a manual check against the module.
- **A reservation is not guaranteed.** Google can still refuse it when the zone has no capacity, as we saw in PRE. The tool tells us what to ask for; getting it is a separate step.
