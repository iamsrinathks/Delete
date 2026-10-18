output "reservations" {
  description = "Reservation details keyed by input key."
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

# Ready to feed GKE node pools:
# reservation_affinity {
#   consume_reservation_type = "SPECIFIC_RESERVATION"
#   key                      = "compute.googleapis.com/reservation-name"
#   values                   = [module.reservations.reservation_names["l4-pool"]]
# }
output "reservation_names" {
  description = "Map of input key => reservation name, for GKE reservation_affinity."
  value       = { for k, r in google_compute_reservation.node_pool : k => r.name }
}

output "shared_reservation_paths" {
  description = "Fully qualified names for consuming a shared reservation from another project."
  value = {
    for k, r in google_compute_reservation.node_pool :
    k => "projects/${r.project}/reservations/${r.name}"
  }
}
