terraform {
  required_version = ">= 1.5.0"

  required_providers {
    # delete_at_time, delete_after_duration, source_instance_template and
    # reservation_sharing_policy were added in hashicorp/google 6.44.0
    google = {
      source  = "hashicorp/google"
      version = ">= 6.44.0, < 8.0.0"
    }
  }
}
