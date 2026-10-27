#!/usr/bin/env python3
"""
gke_node_peaks.py - measure GKE node counts per node pool, zone and machine
shape, as input for sizing Compute Engine reservations.

  Step 1  Live inventory    node pools (gcloud container), node VMs
                            (gcloud compute), optional kubectl cross-check.
  Step 2  Historical peaks  Cloud Monitoring per-node metric, counted per
                            alignment window and mapped to pool / zone / shape.
                            Reports MAX, P95, mean - with and without node
                            upgrade windows (surge nodes).
  Step 3  Autoscaler logs   cluster-autoscaler-visibility scaleUp / scaleDown /
                            failed scale-ups / noScaleUp per pool and zone.

Read-only: only describe / list / read calls are made.
Needs: Python 3.8+, gcloud (authenticated). kubectl is optional.
IAM (read-only): container.clusters.get, container.operations.list,
compute.instances.list, monitoring.timeSeries.list, logging.logEntries.list
(e.g. roles/container.viewer, compute.viewer, monitoring.viewer, logging.viewer).

Example:
  python3 gke_node_peaks.py --project my-proj --cluster tr1-pre \
      --location europe-west2 --days 90 --kube-context current
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

SCRIPT_VERSION = "2026-10-26.8"
METRIC = "kubernetes.io/node/cpu/allocatable_cores"   # one series per node (k8s_node)
AUTOSCALER_LOG = "container.googleapis.com/cluster-autoscaler-visibility"
ZONE_RE = re.compile(r"^[a-z]+-[a-z]+\d+-[a-z]$")      # e.g. europe-west2-a


# --------------------------------------------------------------------------- helpers
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)


def base(url):
    return url.rstrip("/").split("/")[-1] if url else ""


def iso(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s):
    """RFC3339 (with or without fractional seconds) -> epoch seconds (int)."""
    s = re.sub(r"\.\d+", "", s.strip()).replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fmt_epoch(e):
    return datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if e else ""


def region_of(location):
    return location.rsplit("-", 1)[0] if ZONE_RE.match(location) else location


def loc_flag(location):
    return f"--zone={location}" if ZONE_RE.match(location) else f"--region={location}"


DIMS = ("pool_zone", "shape_zone", "pool", "zone", "shape", "cluster")


def run_json(cmd, allow_fail=False):
    log("$ " + " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        if allow_fail:
            log(f"  ! {cmd[0]} not found - skipped")
            return None
        sys.exit(f"ERROR: {cmd[0]} not found on PATH")
    except subprocess.CalledProcessError as e:
        if allow_fail:
            log(f"  ! command failed - skipped: {e.stderr.strip()[:400]}")
            return None
        sys.exit(f"ERROR running: {' '.join(cmd)}\n{e.stderr}")
    out = res.stdout.strip()
    return json.loads(out) if out else []


def percentile(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    k = max(0, math.ceil(p / 100 * len(s)) - 1)
    return s[k]


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_table(title, rows, cols):
    print(f"\n=== {title} ===")
    if not rows:
        print("(no data)")
        return
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# --------------------------------------------------------------------------- step 1
def describe_pools(a):
    cluster = run_json(["gcloud", "container", "clusters", "describe", a.cluster,
                        loc_flag(a.location), f"--project={a.project}", "--format=json"])
    pools = []
    for p in cluster.get("nodePools", []) or []:
        c = p.get("config", {}) or {}
        accels = ",".join(f'{x.get("acceleratorType")}x{x.get("acceleratorCount")}'
                          for x in (c.get("accelerators") or []))
        lssd = (c.get("localSsdCount")
                or (c.get("ephemeralStorageLocalSsdConfig") or {}).get("localSsdCount")
                or (c.get("localNvmeSsdBlockConfig") or {}).get("localSsdCount") or 0)
        ra = c.get("reservationAffinity") or {}
        auto = p.get("autoscaling") or {}
        up = p.get("upgradeSettings") or {}
        migs = []
        for u in p.get("instanceGroupUrls", []) or []:
            m = re.search(r"/zones/([^/]+)/instanceGroup(?:Manager)?s/([^/]+)$", u)
            if m:
                migs.append((m.group(1), m.group(2)))
        pools.append({
            "pool": p["name"],
            "machine_type": c.get("machineType", ""),
            "min_cpu_platform": c.get("minCpuPlatform", ""),
            "accelerators": accels,
            "local_ssd_count": int(lssd),
            "spot": bool(c.get("spot") or c.get("preemptible")),
            "reservation_affinity": ra.get("consumeReservationType", "UNSPECIFIED"),
            "reservation_values": ",".join(ra.get("values", []) or []),
            "locations": ",".join(p.get("locations", []) or []),
            "autoscaling": bool(auto.get("enabled")),
            "min_per_zone": auto.get("minNodeCount", ""),
            "max_per_zone": auto.get("maxNodeCount", ""),
            "total_min": auto.get("totalMinNodeCount", ""),
            "total_max": auto.get("totalMaxNodeCount", ""),
            "location_policy": auto.get("locationPolicy", ""),
            "upgrade_strategy": up.get("strategy", "SURGE"),
            "max_surge": int(up.get("maxSurge", 0) or 0),
            "max_unavailable": int(up.get("maxUnavailable", 0) or 0),
            "version": p.get("version", ""),
            "migs": migs,
        })
    return pools


def list_vms(a, region):
    vms = run_json(["gcloud", "compute", "instances", "list", f"--project={a.project}",
                    f"--filter=labels.goog-k8s-cluster-name={a.cluster}", "--format=json"])
    out = []
    for v in vms or []:
        zone = base(v.get("zone"))
        if not zone.startswith(region + "-"):
            continue  # same cluster name in another region
        labels = v.get("labels", {}) or {}
        sched = v.get("scheduling", {}) or {}
        scratch = [d for d in v.get("disks", []) or [] if d.get("type") == "SCRATCH"]
        out.append({
            "name": v["name"],
            "zone": zone,
            "pool": labels.get("goog-k8s-node-pool-name", ""),
            "machine_type": base(v.get("machineType")),
            "status": v.get("status", ""),
            "min_cpu_platform": v.get("minCpuPlatform", ""),
            "cpu_platform": v.get("cpuPlatform", ""),
            "accelerators": ",".join(f'{base(g.get("acceleratorType"))}x{g.get("acceleratorCount")}'
                                     for g in (v.get("guestAccelerators") or [])),
            "local_ssd_count": len(scratch),
            "local_ssd_interface": ",".join(sorted({d.get("interface", "") for d in scratch})),
            "provisioning": "SPOT" if (sched.get("provisioningModel") == "SPOT"
                                       or sched.get("preemptible")) else "STANDARD",
            "vm_reservation_affinity": (v.get("reservationAffinity") or {}).get(
                "consumeReservationType", ""),
        })
    return out


def list_k8s_nodes(a):
    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    if a.kube_context and a.kube_context != "current":
        cmd += ["--context", a.kube_context]
    d = run_json(cmd, allow_fail=True)
    if not d:
        return None
    nodes = []
    for n in d.get("items", []):
        lb = n["metadata"].get("labels", {})
        ready = next((c["status"] for c in n.get("status", {}).get("conditions", [])
                      if c["type"] == "Ready"), "Unknown")
        nodes.append({"name": n["metadata"]["name"],
                      "pool": lb.get("cloud.google.com/gke-nodepool", ""),
                      "zone": lb.get("topology.kubernetes.io/zone", ""),
                      "machine_type": lb.get("node.kubernetes.io/instance-type", ""),
                      "ready": ready,
                      "unschedulable": n.get("spec", {}).get("unschedulable", False)})
    return nodes


def shape_key(pool, vms_by_pool):
    """Reservation-relevant shape: machine type, min CPU platform, GPUs, Local SSD
    (count + interface), provisioning model."""
    ifaces = sorted({v["local_ssd_interface"] for v in vms_by_pool.get(pool["pool"], [])
                     if v["local_ssd_interface"]})
    lssd = f'{pool["local_ssd_count"]}{"/" + "+".join(ifaces) if ifaces else ""}'
    # Reservations match the VM's machine type as Compute Engine reports it. Compute Engine
    # can report a custom shape under a predefined name (n2-custom-32-32768 -> n2-highcpu-32),
    # so prefer the most common live VM type and record any difference.
    live_types = Counter(v["machine_type"] for v in vms_by_pool.get(pool["pool"], []))
    mtype = live_types.most_common(1)[0][0] if live_types else pool["machine_type"]
    pool["vm_machine_type"] = mtype
    if live_types and mtype != pool["machine_type"]:
        log(f'  ! pool {pool["pool"]}: config machine type {pool["machine_type"]} but VMs report '
            f'{mtype} - shape uses {mtype} (what a reservation must match)')
    return (f'{mtype}|cpu={pool["min_cpu_platform"] or "default"}'
            f'|gpu={pool["accelerators"] or "none"}|lssd={lssd}'
            f'|{"SPOT" if pool["spot"] else "STANDARD"}')


# --------------------------------------------------------------------------- step 1b
def shape_match_key(shape):
    """(machine type, min CPU platform, gpus, local SSD count) from a shape string."""
    p = (shape or "").split("|")
    if len(p) < 5:
        return None
    cpu = p[1][4:] if p[1].startswith("cpu=") else ""
    gpu = p[2][4:] if p[2].startswith("gpu=") else ""
    lssd = p[3][5:].split("/")[0] if p[3].startswith("lssd=") else "0"
    return (p[0], "" if cpu in ("default", "Automatic") else cpu,
            "" if gpu == "none" else gpu, int(lssd or 0))


def collect_capacity(a, region, vms):
    """Existing reservations, regional quotas and machine specs (all read-only)."""
    out = {"reservations": [], "quotas": [], "machine_specs": {}}
    res = run_json(["gcloud", "compute", "reservations", "list", f"--project={a.project}",
                    "--format=json"], allow_fail=True) or []
    for r in res:
        zone = base(r.get("zone"))
        if not zone.startswith(region + "-"):
            continue
        sr = r.get("specificReservation", {}) or {}
        ip = sr.get("instanceProperties", {}) or {}
        accels = ",".join(f'{base(g.get("acceleratorType"))}x{g.get("acceleratorCount")}'
                          for g in (ip.get("guestAccelerators") or []))
        lssd = ip.get("localSsds") or []
        cpu = ip.get("minCpuPlatform", "") or ""
        count, in_use = int(sr.get("count", 0) or 0), int(sr.get("inUseCount", 0) or 0)
        assured = sr.get("assuredCount")
        out["reservations"].append({
            "name": r.get("name", ""), "zone": zone, "status": r.get("status", ""),
            "machine_type": base(ip.get("machineType", "")), "min_cpu_platform": cpu,
            "accelerators": accels, "local_ssd_count": len(lssd),
            "count": count, "in_use": in_use, "idle": max(0, count - in_use),
            "assured_count": int(assured) if assured not in (None, "") else "",
            "specific_required": bool(r.get("specificReservationRequired")),
            "share_type": (r.get("shareSettings") or {}).get("shareType", "LOCAL"),
            "key": list((base(ip.get("machineType", "")), "" if cpu.lower() == "automatic" else cpu,
                         accels, len(lssd))),
        })
    reg = run_json(["gcloud", "compute", "regions", "describe", region, f"--project={a.project}",
                    "--format=json"], allow_fail=True) or {}
    for q in reg.get("quotas", []) or []:
        lim, use = float(q.get("limit", 0) or 0), float(q.get("usage", 0) or 0)
        if lim > 1e15:              # "unlimited" sentinel (9.22e18) - not useful in the report
            continue
        out["quotas"].append({"metric": q.get("metric", ""), "limit": lim, "usage": use,
                              "available": max(0.0, lim - use)})
    zone_for = {}
    for v in vms:
        zone_for.setdefault(v["machine_type"], v["zone"])
    for r in out["reservations"]:
        zone_for.setdefault(r["machine_type"], r["zone"])
    for mt, z in sorted(zone_for.items()):
        if not mt:
            continue
        d = run_json(["gcloud", "compute", "machine-types", "describe", mt, f"--zone={z}",
                      f"--project={a.project}", "--format=json"], allow_fail=True)
        if d:
            out["machine_specs"][mt] = {"vcpus": int(d.get("guestCpus", 0) or 0),
                                        "memory_gib": round(int(d.get("memoryMb", 0) or 0) / 1024, 2)}
    return out


def attach_reservations(sizing, reservations, pools, pool_shape):
    """Adds existing reserved / in-use counts to each shape+zone row and marks which
    pools each reservation can serve (exact shape match in the same zone)."""
    by = defaultdict(list)
    for r in reservations:
        by[(tuple(r["key"]), r["zone"])].append(r)
    for row in sizing:
        k = shape_match_key(row["shape"])
        hits = by.get((k, row["zone"]), []) if k else []
        row["existing_reserved"] = sum(r["count"] for r in hits)
        row["existing_in_use"] = sum(r["in_use"] for r in hits)
        row["existing_names"] = ",".join(r["name"] for r in hits)
        if any(r["specific_required"] for r in hits):
            row["note"] = "; ".join(x for x in (row.get("note", ""),
                                    "existing reservation is SPECIFIC - pools need specific affinity") if x)
    for r in reservations:
        k = tuple(r["key"])
        r["matched_pools"] = ",".join(sorted(p["pool"] for p in pools
                                             if shape_match_key(pool_shape[p["pool"]]) == k
                                             and r["zone"] in (p["locations"] or r["zone"])))
        r["key"] = "|".join(str(x) for x in r["key"])


# --------------------------------------------------------------------------- step 2
class Token:
    def __init__(self):
        self.tok, self.t = None, 0

    def get(self, force=False):
        if force or not self.tok or time.time() - self.t > 1800:
            self.tok = subprocess.run(["gcloud", "auth", "print-access-token"],
                                      capture_output=True, text=True, check=True).stdout.strip()
            self.t = time.time()
        return self.tok


def setup_proxy():
    """urllib does not read gcloud's proxy settings. Reuse them (or HTTPS_PROXY) if set."""
    if os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"):
        log(f"  using HTTPS_PROXY={os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')}")
        return
    def cfg(key):
        try:
            return subprocess.run(["gcloud", "config", "get-value", key], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    addr, port = cfg("proxy/address"), cfg("proxy/port")
    if addr:
        user, pwd = cfg("proxy/username"), cfg("proxy/password")
        auth = f"{urllib.parse.quote(user)}:{urllib.parse.quote(pwd)}@" if user else ""
        proxy = f"http://{auth}{addr}{':' + port if port else ''}"
        urllib.request.install_opener(urllib.request.build_opener(
            urllib.request.ProxyHandler({"https": proxy, "http": proxy})))
        log(f"  using gcloud proxy {addr}{':' + port if port else ''}")


def http_get_json(url, tok, timeout=90):
    force = False
    for attempt in range(6):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok.get(force)}"})
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            data = json.loads(body.decode())
            data["_bytes"], data["_secs"] = len(body), round(time.time() - t0, 1)
            return data
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500]
            if e.code == 401 and attempt == 0:
                force = True
                continue
            if e.code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                log(f"  ! HTTP {e.code}, retrying in {wait}s")
                time.sleep(wait)
                continue
            sys.exit(f"ERROR: Monitoring API HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            wait = 2 ** attempt
            log(f"  ! network error {e.reason}, retrying in {wait}s")
            time.sleep(wait)
        except (TimeoutError, ConnectionError, OSError) as e:   # read timeouts, resets
            wait = 2 ** attempt
            log(f"  ! timeout/connection error ({e}), retrying in {wait}s")
            time.sleep(wait)
    sys.exit("ERROR: Monitoring API failed after retries - check network/proxy access to "
             "monitoring.googleapis.com (see --chunk-days / HTTPS_PROXY)")


def preflight(a, region, tok):
    """Tiny 15-minute query: proves API reachability and auth before the long run."""
    end = datetime.now(timezone.utc).replace(microsecond=0)
    params = {"filter": f'metric.type="{METRIC}" AND resource.type="k8s_node" '
                        f'AND resource.labels.cluster_name="{a.cluster}"',
              "interval.startTime": iso(end - timedelta(minutes=15)), "interval.endTime": iso(end),
              "view": "HEADERS"}
    url = (f"https://monitoring.googleapis.com/v3/projects/{a.project}/timeSeries?"
           + urllib.parse.urlencode(params))
    log("  preflight: 15-minute test query to monitoring.googleapis.com ...")
    data = http_get_json(url, tok, timeout=30)
    log(f"  preflight OK in {data.get('_secs', 0)}s: {len(data.get('timeSeries', []) or [])} node series now")


def group_keys(pool, zone, shape):
    return (("pool_zone", (pool, zone)), ("shape_zone", (shape, zone)), ("pool", (pool,)),
            ("zone", (zone,)), ("shape", (shape,)), ("cluster", ("ALL",)))


def fetch_counts(a, region, resolve, pool_shape):
    """Streams Cloud Monitoring per-node series and counts nodes per window and group.
    Memory grows with groups x windows, not nodes x windows."""
    tok = Token()
    setup_proxy()
    preflight(a, region, tok)
    al = a.alignment
    t_start, n_chunks = time.time(), 0
    end_e = int(time.time()) // al * al                       # windows on alignment boundaries
    start_e = end_e - a.days * 86400 // al * al
    chunk = max(al, a.chunk_days * 86400 // al * al)
    flt = (f'metric.type="{METRIC}" AND resource.type="k8s_node" '
           f'AND resource.labels.cluster_name="{a.cluster}" '
           f'AND resource.labels.location=starts_with("{region}")')
    series = {d: defaultdict(lambda: defaultdict(int)) for d in DIMS}
    windows, node_map, node_loc = set(), {}, {}
    lo = start_e
    while lo < end_e:
        hi = min(lo + chunk, end_e)
        cs = datetime.fromtimestamp(lo, timezone.utc)
        ce = datetime.fromtimestamp(hi, timezone.utc)
        n_chunks += 1
        total_chunks = math.ceil((end_e - start_e) / chunk)
        log(f"  monitoring chunk {n_chunks}/{total_chunks}: {iso(cs)} -> {iso(ce)}")
        chunk_seen = {}                                     # node -> windows in this chunk
        page, page_no = None, 0
        while True:
            params = {"filter": flt,
                      "interval.startTime": iso(cs), "interval.endTime": iso(ce),
                      "aggregation.alignmentPeriod": f"{a.alignment}s",
                      "aggregation.perSeriesAligner": "ALIGN_MAX",
                      "view": "FULL", "pageSize": str(min(100000, max(1000, a.page_size)))}
            if page:
                params["pageToken"] = page
            url = (f"https://monitoring.googleapis.com/v3/projects/{a.project}/timeSeries?"
                   + urllib.parse.urlencode(params))
            data = http_get_json(url, tok)
            page_no += 1
            n_series = len(data.get("timeSeries", []) or [])
            n_points = sum(len(t.get("points", []) or []) for t in data.get("timeSeries", []) or [])
            log(f"    page {page_no}: {n_series} series, {n_points} points, "
                f"{data.get('_bytes', 0) / 1e6:.1f} MB in {data.get('_secs', 0)}s")
            for ts in data.get("timeSeries", []) or []:
                rl = ts.get("resource", {}).get("labels", {})
                node = rl.get("node_name")
                if not node:
                    continue
                node_loc.setdefault(node, rl.get("location", ""))
                if node not in node_map:
                    node_map[node] = resolve(node, node_loc[node])
                pool, zone, _ = node_map[node]
                keys = group_keys(pool, zone, pool_shape.get(pool, "UNKNOWN"))
                seen = chunk_seen.setdefault(node, set())   # guards duplicate series per node
                for p in ts.get("points", []) or []:
                    w = parse_ts(p["interval"]["endTime"])
                    if w > hi or (w <= lo and lo != start_e) or w in seen:
                        continue
                    seen.add(w)
                    windows.add(w)
                    for dim, key in keys:
                        series[dim][key][w] += 1
            page = data.get("nextPageToken")
            if not page:
                break
        lo = hi
        done = (lo - start_e) / max(1, end_e - start_e)
        el = time.time() - t_start
        log(f"  progress {done:.0%}, elapsed {el:.0f}s, ETA ~{el / done * (1 - done):.0f}s, "
            f"nodes seen {len(node_map)}")
    return series, sorted(windows), node_map, node_loc


def build_resolver(pools, vms):
    """node name -> (pool, zone, method). Order: live VM labels, MIG name prefix,
    then a name heuristic for nodes from pools/MIGs that no longer exist."""
    by_name = {v["name"]: (v["pool"], v["zone"]) for v in vms}
    mig_prefix = {}
    for p in pools:
        for zone, mig in p["migs"]:
            mig_prefix[mig[:-4] if mig.endswith("-grp") else mig] = (p["pool"], zone)
    pool_names = sorted((p["pool"] for p in pools), key=len, reverse=True)

    def resolve(node, mon_loc=""):
        if node in by_name:
            return by_name[node] + ("live",)
        pre = node.rsplit("-", 1)[0]
        if pre in mig_prefix:
            return mig_prefix[pre] + ("mig",)
        zone = mon_loc if ZONE_RE.match(mon_loc or "") else "unknown-zone"
        core = node[4:] if node.startswith("gke-") else node
        parts = core.rsplit("-", 2)               # <cluster>-<pool>, <hash>, <suffix>
        if len(parts) == 3:
            for pn in pool_names:                 # node names truncate long pool names
                for k in range(len(pn), 3, -1):
                    if parts[0].endswith("-" + pn[:k]):
                        return (pn, zone, "heuristic")
        return ("UNMAPPED", zone, "unmapped")

    return resolve


def summarise(by_w, windows, excluded):
    vals = [by_w.get(w, 0) for w in windows]
    peak = max(vals) if vals else 0
    peak_w = next((w for w in windows if by_w.get(w, 0) == peak), None)
    kept = [w for w in windows if w not in excluded]
    ex_vals = [by_w.get(w, 0) for w in kept]
    ex_peak = max(ex_vals) if ex_vals else 0
    ex_peak_w = next((w for w in kept if by_w.get(w, 0) == ex_peak), None)
    return {"peak": peak, "peak_time": fmt_epoch(peak_w),
            "peak_excl_upgrades": ex_peak, "peak_excl_upgrades_time": fmt_epoch(ex_peak_w),
            "p95": percentile(vals, 95), "mean": round(sum(vals) / len(vals), 2) if vals else 0,
            "min": min(vals) if vals else 0}


def upgrade_windows(a, windows):
    """Windows overlapping UPGRADE_NODES operations on this cluster (surge nodes)."""
    ops = run_json(["gcloud", "container", "operations", "list", f"--project={a.project}",
                    loc_flag(a.location), "--filter=operationType=UPGRADE_NODES",
                    "--format=json"], allow_fail=True) or []
    spans, rows = [], []
    for op in ops:
        tl = op.get("targetLink", "") or ""
        if f"/clusters/{a.cluster}/" not in tl + "/":
            continue
        try:
            s = parse_ts(op["startTime"])
            e = parse_ts(op["endTime"]) if op.get("endTime") else int(time.time())
        except (KeyError, ValueError):
            continue
        spans.append((s - a.alignment, e + a.alignment))
        rows.append({"operation": op.get("name", ""), "node_pool": base(tl) if "/nodePools/" in tl else "",
                     "status": op.get("status", ""), "start": fmt_epoch(s), "end": fmt_epoch(e)})
    excluded = {w for w in windows for s, e in spans if s <= w <= e}
    return excluded, rows


# --------------------------------------------------------------------------- step 3
def autoscaler_events(a):
    flt = (f'resource.type="k8s_cluster" AND resource.labels.cluster_name="{a.cluster}" '
           f'AND resource.labels.location="{a.location}" AND log_id("{AUTOSCALER_LOG}") '
           f'AND (jsonPayload.decision.scaleUp:* OR jsonPayload.decision.scaleDown:* '
           f'OR jsonPayload.resultInfo:* OR jsonPayload.noDecisionStatus.noScaleUp:*)')
    entries = run_json(["gcloud", "logging", "read", flt, f"--project={a.project}",
                        f"--freshness={a.days}d", f"--limit={a.log_limit}", "--format=json"],
                       allow_fail=True)
    if entries is None:
        return None, None
    if len(entries) >= a.log_limit:
        log(f"  ! hit --log-limit={a.log_limit}; older events not read - raise the limit")

    agg = defaultdict(lambda: {"scale_up_events": 0, "nodes_requested": 0, "max_single_request": 0,
                               "nodes_scaled_down": 0, "failed_scale_ups": Counter(),
                               "no_scale_up_rejections": Counter(), "first": None, "last": None})
    event_targets = {}
    results = []

    def touch(key, ts):
        r = agg[key]
        r["first"] = ts if r["first"] is None else min(r["first"], ts)
        r["last"] = ts if r["last"] is None else max(r["last"], ts)

    for e in entries:
        jp = e.get("jsonPayload", {}) or {}
        ts = parse_ts(e.get("timestamp", iso(datetime.now(timezone.utc))))
        dec = jp.get("decision", {}) or {}
        su = dec.get("scaleUp")
        if su:
            keys = []
            for im in su.get("increasedMigs", []) or []:
                mig = im.get("mig", {}) or {}
                key = (mig.get("nodepool", "?"), mig.get("zone", "?"))
                req = int(im.get("requestedNodes", 0) or 0)
                agg[key]["scale_up_events"] += 1
                agg[key]["nodes_requested"] += req
                agg[key]["max_single_request"] = max(agg[key]["max_single_request"], req)
                touch(key, ts)
                keys.append(key)
            if dec.get("eventId"):
                event_targets[dec["eventId"]] = keys
        sd = dec.get("scaleDown")
        if sd:
            for n in sd.get("nodesToBeRemoved", []) or []:
                mig = (n.get("node", {}) or {}).get("mig", {}) or {}
                key = (mig.get("nodepool", "?"), mig.get("zone", "?"))
                agg[key]["nodes_scaled_down"] += 1
                touch(key, ts)
        for r in (jp.get("resultInfo", {}) or {}).get("results", []) or []:
            em = r.get("errorMsg")
            if em:
                results.append((r.get("eventId"), em.get("messageId", "unknown"), ts))
        nsu = (jp.get("noDecisionStatus", {}) or {}).get("noScaleUp")
        if nsu:
            for rm in nsu.get("rejectedMigs", []) or []:
                mig = rm.get("mig", {}) or {}
                key = (mig.get("nodepool", "?"), mig.get("zone", "?"))
                reason = (rm.get("reason", {}) or {}).get("messageId", "unknown")
                agg[key]["no_scale_up_rejections"][reason] += 1
                touch(key, ts)

    for event_id, msg, ts in results:
        for key in event_targets.get(event_id, [("?", "?")]):
            agg[key]["failed_scale_ups"][msg] += 1
            touch(key, ts)

    rows = []
    for (pool, zone), r in sorted(agg.items()):
        rows.append({"pool": pool, "zone": zone,
                     "scale_up_events": r["scale_up_events"], "nodes_requested": r["nodes_requested"],
                     "max_single_request": r["max_single_request"],
                     "nodes_scaled_down": r["nodes_scaled_down"],
                     "failed_scale_ups": "; ".join(f"{k}={v}" for k, v in r["failed_scale_ups"].items()),
                     "no_scale_up_rejections": "; ".join(
                         f"{k}={v}" for k, v in r["no_scale_up_rejections"].items()),
                     "first_event": fmt_epoch(r["first"]), "last_event": fmt_epoch(r["last"])})
    return rows, len(entries)


# --------------------------------------------------------------------------- sizing
def sizing_rows(shape_zone_stats, pools, pool_shape, a):
    out = []
    for (shape, zone), st in sorted(shape_zone_stats.items()):
        members = [p for p in pools if pool_shape[p["pool"]] == shape
                   and (not p["locations"] or zone in p["locations"].split(","))]
        specific = any(p["reservation_affinity"] == "SPECIFIC_RESERVATION" for p in members)
        surge = sum(p["max_surge"] for p in members) if (specific or a.include_surge) else 0
        basis = st["peak"] if getattr(a, "basis", "peak") == "peak" else (st["peak_excl_upgrades"] or st["peak"])
        factor = 1 + a.growth_pct / 100
        up = math.ceil(round(basis * factor, 6))      # round() avoids 10 x 1.1 -> 11.000000000000002 -> 12
        sized = up + surge
        formula = f"{basis} x {factor:.2f} = {basis * factor:.2f} -> {up}" + (f" + {surge} surge = {sized}" if surge else "")
        note = []
        if shape == "UNKNOWN":
            note.append("unmapped nodes - not sizeable")
        if "|SPOT" in shape:
            note.append("Spot VMs cannot consume reservations")
        if specific:
            note.append("specific affinity: history is capped by the current reservation")
        if any(p["upgrade_strategy"] == "BLUE_GREEN" for p in members):
            note.append("blue-green upgrade needs ~2x nodes during upgrade - surge not modelled")
        if len({p["reservation_affinity"] for p in members}) > 1:
            note.append("pools with different affinities share this shape - review surge")
        out.append({"shape": shape, "zone": zone, "pools": ",".join(p["pool"] for p in members),
                    "peak_excl_upgrades": st["peak_excl_upgrades"], "peak": st["peak"],
                    "growth_pct": a.growth_pct, "surge_headroom": surge, "formula": formula,
                    "sizing_input": sized if shape != "UNKNOWN" and "|SPOT" not in shape else "",
                    "note": "; ".join(note)})
    return out


# --------------------------------------------------------------------------- html
HTML_MARKER = "/*__REPORT_DATA__*/null"
DEFAULT_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gke_capacity_report.html")


def write_html(reports, template, out_path):
    """Embed one or more report_data.json objects into the HTML template."""
    if not os.path.exists(template):
        log(f"  ! HTML template not found ({template}) - report.html skipped")
        return False
    with open(template) as f:
        html = f.read()
    if HTML_MARKER not in html:
        log("  ! HTML template has no data marker - report.html skipped")
        return False
    data = json.dumps(reports, separators=(",", ":")).replace("</", "<\\/")
    with open(out_path, "w") as f:
        f.write(html.replace(HTML_MARKER, data, 1))
    return True


def build_html_cmd(argv):
    ap = argparse.ArgumentParser(prog="gke_node_peaks.py build-html",
                                 description="Merge report_data.json files into one HTML page.")
    ap.add_argument("files", nargs="+", help="report_data.json files (one per cluster)")
    ap.add_argument("--out", default="gke-capacity-report.html")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    a = ap.parse_args(argv)
    reports = []
    for p in a.files:
        with open(p) as f:
            reports.append(json.load(f))
    if write_html(reports, a.template, a.out):
        print(f"Wrote {os.path.abspath(a.out)} with {len(reports)} cluster(s)")


# --------------------------------------------------------------------------- main
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "build-html":
        return build_html_cmd(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--location", required=True, help="cluster region (e.g. europe-west2) or zone")
    ap.add_argument("--days", type=int, default=30, help="history window (default 30)")
    ap.add_argument("--alignment", type=int, default=300,
                    help="seconds per counting window (default 300; min 60)")
    ap.add_argument("--chunk-days", type=int, default=1, help="days per Monitoring API request (default 1)")
    ap.add_argument("--page-size", type=int, default=100000,
                    help="max data POINTS per Monitoring API page with view=FULL "
                         "(API maximum and default 100000)")
    ap.add_argument("--log-limit", type=int, default=100000, help="max autoscaler log entries")
    ap.add_argument("--kube-context", help="run kubectl cross-check ('current' or a context name)")
    ap.add_argument("--growth-pct", type=float, default=0.0, help="growth %% added in sizing input")
    ap.add_argument("--exclude-window", action="append", default=[], metavar="START,END",
                    help="UTC time range to drop from the history, e.g. an outage: "
                         "2026-10-20T00:00Z,2026-10-22T00:00Z (repeatable)")
    ap.add_argument("--basis", choices=["peak", "peak_excl_upgrades"], default="peak",
                    help="value the headroom %% is applied to (default: peak)")
    ap.add_argument("--include-surge", action="store_true",
                    help="add maxSurge headroom even for non-specific affinity pools")
    ap.add_argument("--skip-monitoring", action="store_true")
    ap.add_argument("--skip-logs", action="store_true")
    ap.add_argument("--skip-capacity", action="store_true",
                    help="skip existing reservations, regional quotas and machine specs")
    ap.add_argument("--out-dir", help="output directory (default ./node-peaks-<cluster>-<ts>)")
    ap.add_argument("--html-template", default=DEFAULT_TEMPLATE,
                    help="HTML template for report.html (default: gke_capacity_report.html next to this script)")
    a = ap.parse_args()
    a.alignment = max(60, a.alignment)
    log(f"gke_node_peaks.py version {SCRIPT_VERSION} (Monitoring pageSize={min(100000, max(1000, a.page_size))} points)")
    region = region_of(a.location)
    out = a.out_dir or f"node-peaks-{a.cluster}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out, exist_ok=True)
    summary = {"project": a.project, "cluster": a.cluster, "location": a.location,
               "days": a.days, "alignment_s": a.alignment, "generated": iso(datetime.now(timezone.utc))}
    report = {"meta": summary, "pools": [], "live": [], "peaks": {}, "sizing": [], "autoscaler": []}

    # ---- step 1
    log("STEP 1: live inventory")
    pools = describe_pools(a)
    vms = list_vms(a, region)
    vms_by_pool = defaultdict(list)
    for v in vms:
        vms_by_pool[v["pool"]].append(v)
    pool_shape = {p["pool"]: shape_key(p, vms_by_pool) for p in pools}
    for p in pools:
        p["shape"] = pool_shape[p["pool"]]
        p["live_nodes"] = len(vms_by_pool.get(p["pool"], []))

    pool_cols = ["pool", "shape", "vm_machine_type", "live_nodes", "locations", "autoscaling", "min_per_zone",
                 "max_per_zone", "total_min", "total_max", "location_policy",
                 "reservation_affinity", "reservation_values", "upgrade_strategy",
                 "max_surge", "max_unavailable", "version"]
    write_csv(f"{out}/node_pools.csv", pools, pool_cols)
    print_table("Node pools", pools, ["pool", "shape", "live_nodes", "max_per_zone",
                                      "reservation_affinity", "max_surge", "max_unavailable"])

    vm_cols = ["name", "pool", "zone", "machine_type", "status", "min_cpu_platform", "cpu_platform",
               "accelerators", "local_ssd_count", "local_ssd_interface", "provisioning",
               "vm_reservation_affinity"]
    write_csv(f"{out}/live_vms.csv", vms, vm_cols)
    live = Counter((v["pool"], v["zone"], v["machine_type"], v["status"]) for v in vms)
    live_rows = [{"pool": k[0], "zone": k[1], "machine_type": k[2], "status": k[3], "vms": n}
                 for k, n in sorted(live.items())]
    write_csv(f"{out}/live_counts_pool_zone.csv", live_rows,
              ["pool", "zone", "machine_type", "status", "vms"])
    print_table("Live VMs by pool / zone / machine type", live_rows,
                ["pool", "zone", "machine_type", "status", "vms"])
    report["pools"] = [{k: v for k, v in p.items() if k != "migs"} for p in pools]
    report["live"] = live_rows
    capacity = {"reservations": [], "quotas": [], "machine_specs": {}}
    if not a.skip_capacity:
        log("STEP 1b: existing reservations, regional quotas, machine specs")
        capacity = collect_capacity(a, region, vms)
        report.update(capacity)
        write_csv(f"{out}/existing_reservations.csv", capacity["reservations"],
                  ["name", "zone", "status", "machine_type", "min_cpu_platform", "accelerators",
                   "local_ssd_count", "count", "in_use", "idle", "assured_count", "specific_required",
                   "share_type"])
        write_csv(f"{out}/regional_quotas.csv", capacity["quotas"], ["metric", "limit", "usage", "available"])
        print_table("Existing reservations in region", capacity["reservations"],
                    ["name", "zone", "machine_type", "count", "in_use", "idle", "specific_required"])
        idle = sum(r["idle"] for r in capacity["reservations"])
        if idle:
            log(f"  ! {idle} reserved VM slot(s) are idle - you pay for them at on-demand rates")
    # Reservations match machine type + min CPU platform (not the actual platform a VM got).
    mixed = [p for p, vs in vms_by_pool.items()
             if len({(v["machine_type"], v["min_cpu_platform"]) for v in vs}) > 1]
    if mixed:
        log(f"  ! pools with mixed machine type / min CPU platform live (mid-change?): {mixed}")
    summary["pools_mixed_shape_live"] = mixed
    summary["live_vm_total"] = len(vms)

    if a.kube_context:
        nodes = list_k8s_nodes(a)
        if nodes is not None:
            vm_names, k8s_names = {v["name"] for v in vms}, {n["name"] for n in nodes}
            xc = {"k8s_nodes": len(nodes), "vms": len(vm_names),
                  "in_k8s_not_vm": sorted(k8s_names - vm_names),
                  "vm_not_in_k8s": sorted(vm_names - k8s_names),
                  "not_ready": sorted(n["name"] for n in nodes if n["ready"] != "True"),
                  "cordoned": sorted(n["name"] for n in nodes if n["unschedulable"])}
            overlap = len(k8s_names & vm_names) / max(1, len(k8s_names))
            xc["name_overlap_pct"] = round(100 * overlap, 1)
            if overlap < 0.5:
                log("  ! kubectl context does not look like this cluster "
                    f"({xc['name_overlap_pct']}% node names match the VMs) - check --kube-context")
            summary["kubectl_crosscheck"] = xc
            write_csv(f"{out}/k8s_nodes.csv", nodes,
                      ["name", "pool", "zone", "machine_type", "ready", "unschedulable"])
            print(f"\nkubectl cross-check: {xc['k8s_nodes']} nodes vs {xc['vms']} VMs; "
                  f"in k8s not VM={len(xc['in_k8s_not_vm'])}, VM not in k8s={len(xc['vm_not_in_k8s'])}, "
                  f"NotReady={len(xc['not_ready'])}, cordoned={len(xc['cordoned'])}")

    # ---- step 2
    if not a.skip_monitoring:
        log("STEP 2: historical peaks from Cloud Monitoring")
        resolve = build_resolver(pools, vms)
        series, windows, node_map, node_loc = fetch_counts(a, region, resolve, pool_shape)
        if a.exclude_window and windows:
            spans = []
            for spec in a.exclude_window:
                try:
                    s_, e_ = spec.split(",", 1)
                    spans.append((parse_ts(s_.strip().replace("Z", ":00Z") if s_.strip().count(":") == 1 else s_.strip()),
                                  parse_ts(e_.strip().replace("Z", ":00Z") if e_.strip().count(":") == 1 else e_.strip())))
                except ValueError:
                    sys.exit(f"ERROR: bad --exclude-window '{spec}' (use START,END in UTC, e.g. 2026-10-20T00:00Z,2026-10-22T00:00Z)")
            before = len(windows)
            windows = [w for w in windows if not any(lo_ <= w <= hi_ for lo_, hi_ in spans)]
            summary["excluded_windows"] = [{"start": fmt_epoch(lo_), "end": fmt_epoch(hi_)} for lo_, hi_ in spans]
            log(f"  excluded {before - len(windows)} window(s) from {len(spans)} --exclude-window range(s)")
        if not windows:
            log("  ! no Monitoring data - check GKE system metrics are enabled and the filter")
        else:
            methods = Counter(m for _, _, m in node_map.values())
            running = sum(1 for v in vms if v["status"] == "RUNNING")
            last_count = series["cluster"][("ALL",)].get(windows[-1], 0)
            summary["history"] = {"first_window": fmt_epoch(windows[0]),
                                  "last_window": fmt_epoch(windows[-1]),
                                  "windows": len(windows), "distinct_nodes": len(node_map),
                                  "node_mapping": dict(methods),
                                  "coverage_check": {"monitoring_nodes_last_window": last_count,
                                                     "running_vms_now": running}}
            if running and abs(last_count - running) > max(1, 0.05 * running):
                log(f"  ! coverage check: Monitoring saw {last_count} nodes in the last window, "
                    f"Compute Engine shows {running} running VMs - investigate before trusting peaks")
            if windows[0] > int(time.time()) - a.days * 86400 + 2 * 86400:
                log(f"  ! history starts {fmt_epoch(windows[0])}, later than requested "
                    f"({a.days} days) - metric retention or a new cluster")
            if methods.get("heuristic") or methods.get("unmapped"):
                log(f"  ! node mapping: {dict(methods)} - see node_mapping.csv")
            write_csv(f"{out}/node_mapping.csv",
                      [{"node": n, "pool": p, "zone": z, "method": m, "monitoring_location": node_loc[n]}
                       for n, (p, z, m) in sorted(node_map.items())],
                      ["node", "pool", "zone", "method", "monitoring_location"])

            excluded, ops_rows = upgrade_windows(a, windows)
            summary["history"]["upgrade_windows_excluded"] = len(excluded)
            write_csv(f"{out}/upgrade_operations.csv", ops_rows,
                      ["operation", "node_pool", "status", "start", "end"])

            stat_cols = ["peak", "peak_time", "peak_excl_upgrades", "peak_excl_upgrades_time",
                         "p95", "mean", "min"]
            names = {"pool_zone": ["pool", "zone"], "shape_zone": ["shape", "zone"],
                     "pool": ["pool"], "zone": ["zone"], "shape": ["shape"], "cluster": ["scope"]}
            shape_zone_stats = {}
            for dim in DIMS:
                rows = []
                for key, by_w in sorted(series[dim].items()):
                    st = summarise(by_w, windows, excluded)
                    if dim == "shape_zone":
                        shape_zone_stats[key] = st
                    rows.append({**dict(zip(names[dim], key)), **st})
                write_csv(f"{out}/peaks_by_{dim}.csv", rows, names[dim] + stat_cols)
                report["peaks"][dim] = rows
                if dim in ("pool_zone", "shape_zone", "cluster"):
                    print_table(f"Peaks by {dim.replace('_', ' / ')}", rows,
                                names[dim] + ["peak", "peak_excl_upgrades", "p95", "mean",
                                              "peak_excl_upgrades_time"])

            ts_rows = [{"window_utc": fmt_epoch(w), "pool": k[0], "zone": k[1], "nodes": by_w.get(w, 0)}
                       for k, by_w in sorted(series["pool_zone"].items()) for w in windows]
            write_csv(f"{out}/timeseries_pool_zone.csv", ts_rows, ["window_utc", "pool", "zone", "nodes"])

            sz = sizing_rows(shape_zone_stats, pools, pool_shape, a)
            if capacity["reservations"] or not a.skip_capacity:
                attach_reservations(sz, capacity["reservations"], pools, pool_shape)
                report["reservations"] = capacity["reservations"]
            report["sizing"] = sz
            write_csv(f"{out}/sizing_input_shape_zone.csv", sz,
                      ["shape", "zone", "pools", "peak_excl_upgrades", "peak", "growth_pct",
                       "surge_headroom", "formula", "sizing_input", "existing_reserved",
                       "existing_in_use", "existing_names", "note"])
            if capacity["reservations"]:
                write_csv(f"{out}/existing_reservations.csv", capacity["reservations"],
                          ["name", "zone", "status", "machine_type", "min_cpu_platform", "accelerators",
                           "local_ssd_count", "count", "in_use", "idle", "assured_count",
                           "specific_required", "share_type", "matched_pools"])
            print_table(f"Sizing input per shape / zone ({a.basis} x (1 + {a.growth_pct:g}%) + surge)", sz,
                        ["shape", "zone", "peak", "peak_excl_upgrades", "formula", "sizing_input",
                         "existing_reserved", "existing_in_use"])

    # ---- step 3
    if not a.skip_logs:
        log("STEP 3: cluster autoscaler visibility logs")
        rows, n = autoscaler_events(a)
        if rows is not None:
            summary["autoscaler_log_entries"] = n
            cols = ["pool", "zone", "scale_up_events", "nodes_requested", "max_single_request",
                    "nodes_scaled_down", "failed_scale_ups", "no_scale_up_rejections",
                    "first_event", "last_event"]
            write_csv(f"{out}/autoscaler_events_pool_zone.csv", rows, cols)
            report["autoscaler"] = rows
            print_table("Autoscaler activity by pool / zone", rows,
                        ["pool", "zone", "scale_up_events", "nodes_requested", "max_single_request",
                         "nodes_scaled_down", "failed_scale_ups"])

    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(f"{out}/report_data.json", "w") as f:
        json.dump(report, f, indent=2)
    if write_html([report], a.html_template, f"{out}/report.html"):
        log(f"  report page: {os.path.abspath(out)}/report.html")
    print(f"\nOutputs written to: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
