# PRE Reservation Apply – Results and Action Plan

Oct 28, 2026 · @S

In PRE, 17 of our 39 reservation requests went through and 22 failed because Google did not have spare capacity in those zones at that moment. The good thing is that our running nodes are not affected at all; what we have lost for now is only the guarantee for future scale-ups, upgrades and node repairs.

## At a glance

We applied reservations for two projects (svp01 and svp02), each with one cluster in europe-west2 (EW2) and one in europe-west1 (EW1). Roughly 8,848 vCPU of what we asked for is still not reserved, and most of that gap is in EW2.

| Region | Zones | Requests | Succeeded | Failed | Not reserved (approx. vCPU) |
| --- | --- | --- | --- | --- | --- |
| EW2 | a, b, c | 24 | 9 | 15 | 6,752 |
| EW1 | b, c, d | 15 | 8 | 7 | 2,096 |
| Total |  | 39 | 17 | 22 | 8,848 |

The vCPU figures are worked out from the instance counts in our tfvars and the vCPUs per machine type, so please treat them as close estimates.

## What succeeded and why

These 17 reservations went through. They worked simply because, at the moment we ran the apply, Google had spare capacity for that machine family in that zone. A zone can be full for one family and still have room for another, which is exactly what we saw.

| Region | Project | Machine type | Zone (VMs reserved) |
| --- | --- | --- | --- |
| EW2 | svp01 | n2-custom-32-73728 | c (27) |
| EW2 | svp01 | n2-custom-32-90112 | b (6), c (7) |
| EW2 | svp01 | n2-custom-64-147456 | c (10) |
| EW2 | svp01 | n2-custom-80-192512 | c (5) |
| EW2 | svp01 | n4-custom-32-73728 | a (6) |
| EW2 | svp02 | n2-standard-32 | c (5) |
| EW2 | svp02 | n2-custom-32-73728 | c (27) |
| EW2 | svp02 | n4-custom-32-73728 | a (13) |
| EW1 | svp01 | n2-custom-32-73728 | c (14), d (13) |
| EW1 | svp01 | n2-custom-80-192512 | c (10), d (9) |
| EW1 | svp01 | n4-custom-32-73728 | b (4) |
| EW1 | svp02 | n4-custom-32-73728 | c (5) |
| EW1 | svp02 | n4-standard-32 | b (5), c (4) |

One thing to keep in mind: a reservation that exists is not automatically useful. Our node pools use ANY affinity, so a node uses a reservation only if its project, zone and machine type match exactly. We still need to confirm the in-use count on each of these (step 2 in the action plan) before we call them done.

## What failed and why

All 22 failures are capacity problems on Google's side, not a quota issue and not a bug in our Terraform. They fall into three types.

| Error type | Where we saw it | What it actually means |
| --- | --- | --- |
| CPU availability (the error has a reason field) | EW2-b n2-custom-64-147456, EW2-b n2-custom-32-73728, EW2-a n2-custom-32-73728 | The zone has no vCPU left for that shape right now. Google's message itself suggested trying the other EW2 zones. |
| General capacity unavailable | Most of the rest: n2 shapes in EW2-a and EW2-b, n4 in EW2-b, n2 in EW1-b, n4 in EW1-c and EW1-d | That shape is simply out of stock in that zone at the moment. Google calls this a resource availability error, and it is separate from quota. |
| RESERVATION\_CAPACITY\_LIMIT\_EXCEEDED | n4-custom-32-73728 in EW2-c (and EW2-b for svp02) | The zone has no spare reservation capacity for the n4 family. Google rejects the request so that we are not charged twice for VMs that are already running there. |

When we put all the results side by side, a few patterns come out clearly. These are our own reading of the results, not a statement from Google.

- In EW2, n2 reservations only went through in zone c and n4 only in zone a. Zone b is the tightest of all: just 1 of 8 requests succeeded there.
- In EW1, n2 failed only in zone b, and n4 never succeeded in zone d.
- The single biggest gap is n2-custom-32-73728 in EW2-a and EW2-b across both projects, roughly 102 VMs or about 3,264 vCPU.
- svp01 and svp02 are fighting for the same zonal capacity, and whichever applies first wins. In EW1-b, for example, svp01 got its n4 reservation and svp02 did not.
- Big requests are all-or-nothing. If a zone can give us 20 VMs and we ask for 26, we get zero.
- Some of our counts are higher than what actually runs in that zone. We asked for about 26 n2 VMs in EW2-b, where far fewer nodes run, so we made the hardest zone even harder to satisfy.

## Action plan

Our plan is to protect what runs today first, then chase the missing capacity with smaller retries, future reservations and a direct conversation with Google. The steps below are written in ASD-STE100 style, so each one is a short, clear instruction.

| # | Action | Why | Output | Status |
| --- | --- | --- | --- | --- |
| 1 | Do not change the running node pools. | A failed reservation does not stop running nodes. | No impact on tenants. | Not started |
| 2 | Run `gcloud compute reservations describe` for each successful reservation. Read the in-use count. | A reservation must match the node VMs exactly. An in-use count of 0 shows a mismatch. | A list of used and idle reservations. | Not started |
| 3 | Make sure the module sets `specific_reservation_required = false`. | Our node pools use ANY affinity. A specific reservation stays idle and we still pay for it. | Confirmed setting. | Not started |
| 4 | Run the capacity tool (v2026-10-28.1) on all four PRE clusters with `--days 30 --surge-mode one`. | The tool gives the peak for each shape in each zone, plus 1 surge VM. | Capacity Reservation Table. | Not started |
| 5 | Replace the even split with the per-zone counts from the table. | Zone b has fewer nodes. A smaller request in zone b has a better chance to succeed. | Updated tfvars. | Not started |
| 6 | Reserve the nodes that run now first. Add headroom after that. | This protects today's workload before growth. | Base reservations in place. | Not started |
| 7 | Split each request of more than 10 VMs into smaller reservations, for example `-a-1` and `-a-2`. | Google advises fewer VMs per request. A small request has a better chance to succeed. | More successful applies. | Not started |
| 8 | Run the apply again for the failed zones every 4 to 6 hours for 3 days. Run svp01 and svp02 at different times. | Capacity changes over time. The two projects compete for the same zone. | Failed reservations retried automatically. | Not started |
| 9 | Read the autoscaler log for `scale.up.error.out.of.resources` in each cluster. | This shows real stockouts, not only reservation failures. | A list of zones with real stockouts. | Not started |
| 10 | Send a future reservation request for n2-custom-32-73728 in EW2-a and EW2-b. | This is the largest gap. Google reviews these requests, typically in about 5 days. | Request ID and approval status. | Not started |
| 11 | For n4 in EW2-c, test an n2 reservation. Do this only if the pool can use n2. | Google's documented fix for the limit error is another machine family. | Decision on the machine family. | Not started |
| 12 | Send the results to the Google account team (TAM). Ask which shapes they can give in each zone. | Failures in 6 zones and 2 regions need a capacity discussion with Google. | Capacity commitment from Google. | Not started |
| 13 | Agree which environment gets capacity first. | PRE uses the same tight zones that PRD needs. | Rule: PRD first, PRE gets a smaller base. | Not started |
| 14 | Record each request in one tracker: zone, shape, count, status and retry date. | The team and Google can see the same status. | Shared tracker. | Not started |

If we can do only one thing tomorrow, it should be step 2. It tells us the 17 reservations we already pay for are really being used before we add any more.

## Sources

- [Troubleshooting reservation creation](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-reservation-creation) (Compute Engine): resource availability errors and RESERVATION\_CAPACITY\_LIMIT\_EXCEEDED, with Google's suggested fixes
- [Troubleshooting resource availability errors](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-resource-availability) (Compute Engine): what the errors mean and the order to try fixes in
- [Consuming reserved zonal resources](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/consuming-reservations) (GKE): ANY and specific affinity, and how regional node pools use zonal reservations
- [About future reservation requests](https://cloud.google.com/compute/docs/future-reservations) (Compute Engine): how future reservations are reviewed and provisioned
- [Choose a reservation type](https://cloud.google.com/compute/docs/instances/choose-reservation-type) (Compute Engine): the typical review time for future reservations
- [Troubleshoot cluster autoscaler not scaling up](https://docs.cloud.google.com/kubernetes-engine/docs/troubleshooting/cluster-autoscaler-scale-up) (GKE): the `scale.up.error.out.of.resources` log query
