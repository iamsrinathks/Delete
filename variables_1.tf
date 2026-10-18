variable "project_id" {
  description = "Project ID where reservations are created."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a valid GCP project ID."
  }
}

variable "name_prefix" {
  description = "Optional prefix added to every reservation name (e.g. 'tr1-pre-')."
  type        = string
  default     = ""
}

variable "node_pool_reservations" {
  description = "Compute Engine reservations for GKE node pools. Map key is a unique per-zone handle; `name` (defaults to the key) is the reservation name and must be identical across zones for a regional node pool."
  type = map(object({
    enabled = optional(bool, false)
    name    = optional(string) # shared name across zones, e.g. l4-inference
    zone    = string

    # Set true when attached to a resource-based CUD (GPUs/most local SSD):
    # such reservations cannot be deleted, resized or given auto-delete.
    commitment_attached = optional(bool, false)

    instance_count = number

    # Either instance_properties-style fields OR source_instance_template
    machine_type         = optional(string)
    min_cpu_platform     = optional(string)
    maintenance_interval = optional(string) # PERIODIC | RECURRENT
    accelerator_type     = optional(string) # short name, e.g. nvidia-l4
    accelerator_count    = optional(number)
    local_ssds = optional(list(object({
      disk_size_gb = number
      interface    = optional(string, "NVME") # NVME | SCSI
    })), [])
    source_instance_template = optional(string)

    specific_reservation_required = optional(bool, true)
    description                   = optional(string)

    # Sharing (cross-project, e.g. shared GKE host/service projects)
    share_type         = optional(string) # LOCAL | SPECIFIC_PROJECTS
    shared_project_ids = optional(list(string), [])
    service_share_type = optional(string) # ALLOW_ALL | DISALLOW_ALL

    # Auto-delete (mutually exclusive)
    delete_at_time = optional(string) # RFC3339, e.g. 2026-12-31T23:59:59Z
    delete_after_duration = optional(object({
      seconds = number
      nanos   = optional(number, 0)
    }))
  }))
  default = {}

  validation {
    condition = alltrue([
      for k, r in var.node_pool_reservations :
      can(regex("^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$", k))
    ])
    error_message = "Reservation keys must be RFC1035 names: lowercase, digits, hyphens, max 63 chars."
  }

  validation {
    condition = alltrue([
      for k, r in var.node_pool_reservations :
      can(regex("^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$", "${var.name_prefix}${coalesce(r.name, k)}"))
    ])
    error_message = "name_prefix + name must be an RFC1035 name of max 63 chars."
  }

  validation {
    condition = length(distinct([
      for k, r in var.node_pool_reservations : "${coalesce(r.name, k)}/${r.zone}"
    ])) == length(var.node_pool_reservations)
    error_message = "Each reservation name can appear only once per zone."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      !r.commitment_attached || (r.delete_at_time == null && r.delete_after_duration == null)
    ])
    error_message = "Reservations attached to a commitment cannot use auto-delete."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      length(r.shared_project_ids) <= 100
    ])
    error_message = "A shared reservation supports at most 100 consumer projects."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      r.machine_type == null || !startswith(r.machine_type, "n4-") || length(r.local_ssds) == 0
    ])
    error_message = "N4 machine types do not support local SSDs."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations : r.instance_count >= 1
    ])
    error_message = "instance_count must be >= 1."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      (r.machine_type != null) != (r.source_instance_template != null)
    ])
    error_message = "Set exactly one of machine_type or source_instance_template."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      (r.accelerator_type == null) == (r.accelerator_count == null)
    ])
    error_message = "accelerator_type and accelerator_count must be set together."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      r.accelerator_type == null || !can(regex("/", r.accelerator_type))
    ])
    error_message = "accelerator_type must be a short name (e.g. nvidia-l4), not a URL."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      !(r.delete_at_time != null && r.delete_after_duration != null)
    ])
    error_message = "delete_at_time and delete_after_duration are mutually exclusive."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      r.delete_at_time == null || can(timeadd(r.delete_at_time, "0s"))
    ])
    error_message = "delete_at_time must be an RFC3339 timestamp."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      r.share_type == null || contains(["LOCAL", "SPECIFIC_PROJECTS"], r.share_type)
    ])
    error_message = "share_type must be LOCAL or SPECIFIC_PROJECTS."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      r.share_type != "SPECIFIC_PROJECTS" || length(r.shared_project_ids) > 0
    ])
    error_message = "shared_project_ids is required when share_type = SPECIFIC_PROJECTS."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      r.service_share_type == null || contains(["ALLOW_ALL", "DISALLOW_ALL"], r.service_share_type)
    ])
    error_message = "service_share_type must be ALLOW_ALL or DISALLOW_ALL."
  }

  validation {
    condition = alltrue([
      for r in var.node_pool_reservations :
      r.maintenance_interval == null || contains(["PERIODIC", "RECURRENT"], r.maintenance_interval)
    ])
    error_message = "maintenance_interval must be PERIODIC or RECURRENT."
  }
}
