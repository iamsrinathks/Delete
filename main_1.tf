locals {
  reservations = {
    for key, cfg in var.node_pool_reservations : key => merge(cfg, {
      # GKE regional/multi-zonal node pools need the SAME name in every zone,
      # so the map key is per-zone (e.g. l4-inference-a) and `name` is shared.
      final_name = "${var.name_prefix}${coalesce(cfg.name, key)}"
    })
    if cfg.enabled
  }
}

resource "google_compute_reservation" "node_pool" {
  for_each = local.reservations

  name        = each.value.final_name
  project     = var.project_id
  zone        = each.value.zone
  description = each.value.description

  specific_reservation_required = each.value.specific_reservation_required

  delete_at_time = each.value.delete_at_time

  dynamic "delete_after_duration" {
    for_each = each.value.delete_after_duration != null ? [each.value.delete_after_duration] : []
    content {
      seconds = tostring(delete_after_duration.value.seconds) # schema type is string
      nanos   = delete_after_duration.value.nanos
    }
  }

  specific_reservation {
    count                    = each.value.instance_count
    source_instance_template = each.value.source_instance_template

    dynamic "instance_properties" {
      for_each = each.value.machine_type != null ? [1] : []
      content {
        machine_type         = each.value.machine_type
        min_cpu_platform     = each.value.min_cpu_platform
        maintenance_interval = each.value.maintenance_interval

        dynamic "guest_accelerators" {
          for_each = each.value.accelerator_type != null ? [1] : []
          content {
            accelerator_type  = each.value.accelerator_type
            accelerator_count = each.value.accelerator_count
          }
        }

        dynamic "local_ssds" {
          for_each = each.value.local_ssds
          content {
            disk_size_gb = local_ssds.value.disk_size_gb
            interface    = local_ssds.value.interface
          }
        }
      }
    }
  }

  dynamic "share_settings" {
    for_each = each.value.share_type != null ? [1] : []
    content {
      share_type = each.value.share_type

      dynamic "project_map" {
        for_each = toset(each.value.shared_project_ids)
        content {
          id         = project_map.value
          project_id = project_map.value
        }
      }
    }
  }

  dynamic "reservation_sharing_policy" {
    for_each = each.value.service_share_type != null ? [1] : []
    content {
      service_share_type = each.value.service_share_type
    }
  }

  # Only these change in place (per Compute Engine docs): instance_count,
  # share type / consumer projects, Vertex sharing policy, auto-delete.
  # Everything else forces a NEW reservation. A specific reservation cannot be
  # deleted, shrunk below in-use, or have a consuming project removed while
  # instances consume it - the API rejects it.
  lifecycle {
    # GPU capacity is scarce: recreating a reservation can lose it.
    # Retire deliberately via the documented procedure.
    prevent_destroy = true

    # Compute Engine may consume/adjust in_use_count; never fight it.
    ignore_changes = [specific_reservation[0].in_use_count]
  }
}
