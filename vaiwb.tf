variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region for Workbench instance"
  type        = string
}

variable "zone" {
  description = "GCP Zone for Workbench instance (must have GPU availability)"
  type        = string
}

variable "instance_name" {
  description = "Name of the Vertex AI Workbench instance"
  type        = string
}

variable "machine_type" {
  description = "Machine type for the Workbench instance (must be GPU-compatible)"
  type        = string
  default     = "n1-standard-8"
}

# ── GPU Configuration ──────────────────────────────────────────────────────────
variable "gpu_type" {
  description = "GPU accelerator type (e.g., nvidia-tesla-t4, nvidia-tesla-a100)"
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  description = "Number of GPU accelerators to attach"
  type        = number
  default     = 1
}

# ── Secure Boot / Shielded VM ──────────────────────────────────────────────────
variable "secure_boot_image_name" {
  description = "Name of the pre-built custom image with self-signed GPU drivers"
  type        = string
  # Example: "ubuntu24-cuda-secureboot-v1"
}

variable "secure_boot_image_project" {
  description = "Project where the custom Secure Boot image resides (defaults to project_id)"
  type        = string
  default     = ""
}

variable "enable_vtpm" {
  description = "Enable Virtual Trusted Platform Module"
  type        = bool
  default     = true
}

variable "enable_integrity_monitoring" {
  description = "Enable Shielded VM integrity monitoring"
  type        = bool
  default     = true
}

# ── Networking ─────────────────────────────────────────────────────────────────
variable "network" {
  description = "VPC network self-link or name"
  type        = string
}

variable "subnet" {
  description = "Subnetwork self-link or name"
  type        = string
}

variable "no_public_ip" {
  description = "Disable public IP (enterprise best practice)"
  type        = bool
  default     = true
}

variable "no_proxy_access" {
  description = "Disable proxy access to the notebook"
  type        = bool
  default     = false
}

# ── Identity & Access ──────────────────────────────────────────────────────────
variable "service_account_email" {
  description = "Service account email to attach to the Workbench instance"
  type        = string
}

# ── Storage ────────────────────────────────────────────────────────────────────
variable "boot_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 150
}

variable "boot_disk_type" {
  description = "Boot disk type"
  type        = string
  default     = "PD_SSD"
}

variable "data_disk_size_gb" {
  description = "Data disk size in GB"
  type        = number
  default     = 200
}

# ── Labels & Tags ──────────────────────────────────────────────────────────────
variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default     = {}
}

variable "tags" {
  description = "Network tags for firewall rules"
  type        = list(string)
  default     = []
}

# ── Image Builder (Phase 1) ────────────────────────────────────────────────────
variable "build_secure_image" {
  description = <<-EOT
    Set to true ONCE to trigger the image build pipeline via null_resource.
    After the image is built, set this to false and provide secure_boot_image_name.
    This prevents unnecessary rebuilds on every plan/apply.
  EOT
  type        = bool
  default     = false
}

variable "base_os_image" {
  description = "Base OS image family for the image builder (e.g., ubuntu-24)"
  type        = string
  default     = "ubuntu-24"
}

variable "builder_zone" {
  description = "Zone to use for the ephemeral image builder VM (can differ from workbench zone)"
  type        = string
  default     = ""
}


---------------------

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Build a custom GCE image with self-signed GPU drivers
#
# This uses Google's open-source cuda_installer.pyz script to:
#   1. Generate a self-signing certificate (MOK)
#   2. Spin up an ephemeral build VM
#   3. Install + sign NVIDIA drivers & CUDA
#   4. Export a GCE image with the cert enrolled as trusted
#   5. Destroy the ephemeral build VM
#
# Run this ONCE (build_secure_image = true), then flip to false and
# reference the produced image name in secure_boot_image_name.
# ─────────────────────────────────────────────────────────────────────────────

locals {
  builder_zone_resolved = var.builder_zone != "" ? var.builder_zone : var.zone
  image_project         = var.secure_boot_image_project != "" ? var.secure_boot_image_project : var.project_id
}

resource "null_resource" "build_secure_boot_gpu_image" {
  count = var.build_secure_image ? 1 : 0

  # Re-trigger only when image name or base OS changes
  triggers = {
    image_name   = var.secure_boot_image_name
    base_image   = var.base_os_image
    builder_zone = local.builder_zone_resolved
  }

  provisioner "local-exec" {
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -euo pipefail

      echo "=== [Phase 1] Downloading cuda_installer.pyz ==="
      curl -sSL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
        --output /tmp/cuda_installer.pyz

      echo "=== [Phase 1] Building Secure Boot compatible GPU image ==="
      python3 /tmp/cuda_installer.pyz build_image \
        --project  ${var.project_id} \
        --vm-zone  ${local.builder_zone_resolved} \
        --base-image ${var.base_os_image} \
        ${var.secure_boot_image_name}

      echo "=== [Phase 1] Image build complete: ${var.secure_boot_image_name} ==="
    EOT
  }
}



-----------------

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0"
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Vertex AI Workbench Instance
#
# Uses the custom image produced in Phase 1 which has:
#   - NVIDIA drivers signed with a self-generated MOK certificate
#   - The MOK certificate enrolled as trusted in the UEFI db
#
# Shielded VM config (Secure Boot + vTPM + Integrity Monitoring) can now
# be enabled together with GPU accelerators without driver signing conflicts.
# ─────────────────────────────────────────────────────────────────────────────

resource "google_workbench_instance" "this" {
  name     = var.instance_name
  location = var.zone
  project  = var.project_id

  # Ensure image is built before instance is created (when build_secure_image = true)
  depends_on = [null_resource.build_secure_boot_gpu_image]

  gce_setup {
    machine_type = var.machine_type

    # ── GPU Accelerator ────────────────────────────────────────────────────────
    accelerators {
      type       = var.gpu_type
      core_count = var.gpu_count
    }

    # ── Custom Secure Boot Compatible Image ────────────────────────────────────
    # This MUST be the image built in Phase 1 with self-signed GPU drivers.
    # Using any other image will cause GPU drivers to fail to load under Secure Boot.
    vm_image {
      project = local.image_project
      name    = var.secure_boot_image_name
      # Use 'name' for a specific versioned image.
      # Alternatively use 'family' if you maintain an image family via packer/image builder.
    }

    # ── Shielded VM: The Key Config ────────────────────────────────────────────
    # All three shielded options enabled — enterprise compliance posture.
    # This works ONLY because the custom image has signed drivers.
    shielded_instance_config {
      enable_secure_boot          = true   # ← The critical flag — NOW safe to enable
      enable_vtpm                 = var.enable_vtpm
      enable_integrity_monitoring = var.enable_integrity_monitoring
    }

    # ── Boot Disk ──────────────────────────────────────────────────────────────
    boot_disk {
      disk_size_gb = var.boot_disk_size_gb
      disk_type    = var.boot_disk_type
      disk_encryption = "GMEK" # Google-managed encryption key (use CMEK for stricter compliance)
    }

    # ── Data Disk ──────────────────────────────────────────────────────────────
    data_disks {
      disk_size_gb = var.data_disk_size_gb
      disk_type    = "PD_SSD"
      disk_encryption = "GMEK"
    }

    # ── Networking: No Public IP (Enterprise Default) ──────────────────────────
    network_interfaces {
      network    = var.network
      subnet     = var.subnet
      nic_type   = "VIRTIO_NET"
      # Omitting access_configs means no external/public IP is assigned
    }

    # ── Service Account ────────────────────────────────────────────────────────
    service_accounts {
      email = var.service_account_email
    }

    # ── Metadata ───────────────────────────────────────────────────────────────
    metadata = {
      # Required for Secure Boot — ensures NVIDIA drivers are NOT reinstalled
      # from unsigned distro packages on startup
      "install-nvidia-driver"    = "false"
      "proxy-mode"               = var.no_proxy_access ? "none" : "service_account"
      "block-project-ssh-keys"   = "true"   # Enforce instance-level SSH keys only
      "enable-oslogin"           = "true"   # OS Login for IAM-based SSH access
      "serial-port-logging-enable" = "true" # Audit boot logs via Cloud Logging
    }

    tags = var.tags

    labels = merge(var.labels, {
      "secure-boot"     = "enabled"
      "gpu-workload"    = "true"
      "managed-by"      = "terraform"
      "shielded-vm"     = "true"
    })
  }

  # Maintenance: GPU VMs must TERMINATE (not migrate) for host maintenance
  # This is mandatory for any VM with GPU accelerators
  instance_owners = []

  disable_proxy_access = var.no_proxy_access
}


-------------

output "workbench_instance_id" {
  description = "Full resource ID of the Workbench instance"
  value       = google_workbench_instance.this.id
}

output "workbench_instance_state" {
  description = "Current state of the Workbench instance"
  value       = google_workbench_instance.this.state
}

output "workbench_proxy_uri" {
  description = "Proxy URI to access the Workbench JupyterLab UI"
  value       = google_workbench_instance.this.proxy_uri
}

output "secure_boot_image_used" {
  description = "The custom Secure Boot compatible image used by this instance"
  value       = var.secure_boot_image_name
}

output "shielded_vm_config" {
  description = "Summary of Shielded VM configuration"
  value = {
    secure_boot          = true
    vtpm                 = var.enable_vtpm
    integrity_monitoring = var.enable_integrity_monitoring
  }
}

-------------

tfvars

# ── Step 1: Build the image (run apply once with build_secure_image = true) ──
build_secure_image     = true
secure_boot_image_name = "ubuntu24-cuda-secureboot-v20250713"
base_os_image          = "ubuntu-24"
builder_zone           = "us-central1-a"

# ── Step 2: Deploy Workbench (set build_secure_image = false after image exists) ──
# build_secure_image   = false

project_id             = "my-enterprise-project"
region                 = "us-central1"
zone                   = "us-central1-a"
instance_name          = "wb-gpu-secure-prod-01"
machine_type           = "n1-standard-8"

gpu_type               = "nvidia-tesla-t4"
gpu_count              = 1

network                = "projects/my-project/global/networks/my-vpc"
subnet                 = "projects/my-project/regions/us-central1/subnetworks/my-subnet"
no_public_ip           = true

service_account_email  = "vertex-wb-sa@my-enterprise-project.iam.gserviceaccount.com"

boot_disk_size_gb      = 150
data_disk_size_gb      = 500

labels = {
  environment = "production"
  team        = "mlops"
  cost-center = "ai-platform"
}
```

---

## Deployment Workflow
```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Run Once (build_secure_image = true)                 │
│                                                                  │
│  terraform apply                                                 │
│      └── null_resource triggers cuda_installer.pyz              │
│              ├── Generates self-signed MOK certificate           │
│              ├── Spins up ephemeral builder VM                   │
│              ├── Installs + signs NVIDIA drivers                 │
│              ├── Bakes into GCE image (MOK cert enrolled)        │
│              └── Destroys ephemeral builder VM (~30 min)         │
│                                                                  │
│  Output: ubuntu24-cuda-secureboot-v20250713 (in GCE Images)     │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Ongoing (build_secure_image = false)                 │
│                                                                  │
│  terraform apply                                                 │
│      └── google_workbench_instance                              │
│              ├── Uses Phase 1 custom image                       │
│              ├── Attaches GPU (nvidia-tesla-t4)                  │
│              ├── enable_secure_boot = true  ✅                   │
│              ├── enable_vtpm        = true  ✅                   │
│              └── enable_integrity_monitoring = true  ✅          │
└─────────────────────────────────────────────────────────────────┘




Key Enterprise Security Considerations
ConcernRecommendationMOK Certificate StorageStore the generated MOK cert/key in Secret Manager after Phase 1, for future driver updatesImage VersioningUse a naming convention with dates (v20250713) and maintain via a GCE Image Family for rolling updatesCMEKReplace GMEK with CMEK + Cloud KMS for regulated workloads (HIPAA, PCI-DSS, FedRAMP)Image Rebuild on Driver UpdateRe-run Phase 1 when NVIDIA releases a new driver version — set build_secure_image = true with a new secure_boot_image_nameOS Loginenable-oslogin = true metadata enforces IAM-based SSH — no static SSH keysNo Public IPno_public_ip = true is the enterprise default; access via IAP Tunnel or VPNIntegrity MonitoringAlerts on unexpected boot sequence changes via Cloud Monitoring / Security Command Center
