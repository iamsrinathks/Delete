output "reservations" {
  description = "Reservation details keyed by the per-zone input key."
  value = {
    for k, r in google_compute_reservation.node_pool : k => {
      id        = r.id
      name      = r.name
      zone      = r.zone
      self_link = r.self_link
      status    = r.status
    }
  }
}

# Shared name per logical reservation (same in every zone) -> feed this to a
# regional GKE node pool's reservation_affinity.values. See examples/node_pool.tf.
output "reservation_names" {
  description = "Map of logical name (without prefix) => reservation name used in all its zones."
  value = {
    for k, r in local.reservations : coalesce(var.node_pool_reservations[k].name, k) => r.final_name...
  }
}

output "reservation_zones" {
  description = "Map of logical name => zones where it is reserved (use for GKE node_locations)."
  value = {
    for k, r in local.reservations : coalesce(var.node_pool_reservations[k].name, k) => r.zone...
  }
}

output "shared_reservation_paths" {
  description = "Full path for consumer projects: projects/<owner>/reservations/<name>."
  value = {
    for k, r in local.reservations :
    coalesce(var.node_pool_reservations[k].name, k) => "projects/${var.project_id}/reservations/${r.final_name}"...
  }
}
