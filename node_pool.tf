# Example for the gke workspace (GKE Standard, regional cluster).
# Reservation affinity is IMMUTABLE on a node pool: decide it at creation.
# Leave the reservation unset for normal pools - omitting the block keeps
# the Compute Engine default (ANY) and avoids a forced pool replacement.

data "tfe_outputs" "prj" {
  organization = var.tfc_org
  workspace    = var.prj_workspace
}

locals {
  # reservation_names / reservation_zones are grouped maps (name => list).
  rsv_name  = try(data.tfe_outputs.prj.nonsensitive_values.reservation_names["l4-inference"][0], null)
  rsv_zones = try(data.tfe_outputs.prj.nonsensitive_values.reservation_zones["l4-inference"], [])
  # Consumer (shared) project: use shared_reservation_paths["l4-inference"][0] instead.
}

resource "google_container_node_pool" "l4_inference" {
  name     = "l4-inference"
  project  = var.gke_project
  cluster  = var.cluster_name
  location = "europe-west2"

  # GPU pools: only zones that actually hold the reservation (same name in each).
  node_locations = local.rsv_zones

  autoscaling {
    total_min_node_count = 0
    total_max_node_count = 6     # <= total reserved count across zones
    location_policy      = "ANY" # follows unused reservations across uneven zones
  }

  upgrade_settings {
    max_surge       = 0 # no spare reserved slot needed for upgrades
    max_unavailable = 1
  }

  node_config {
    machine_type = "g2-standard-8" # must EXACTLY match the reservation

    guest_accelerator {
      type  = "nvidia-l4"
      count = 1
      gpu_driver_installation_config { gpu_driver_version = "DEFAULT" }
    }

    dynamic "reservation_affinity" {
      for_each = local.rsv_name != null ? [1] : []
      content {
        consume_reservation_type = "SPECIFIC_RESERVATION"
        key                      = "compute.googleapis.com/reservation-name"
        values                   = [local.rsv_name]
      }
    }
  }
}
