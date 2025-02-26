#!/bin/bash
set -e  # Exit on any error

echo "Executing post-startup script..."

# Load environment variables
source /etc/profile.d/env.sh || exit 1

# Get Nexus URL dynamically
if [[ -z "$NEXUS_URL" ]]; then
    echo "NEXUS_URL not set, fetching from metadata..."
    NEXUS_URL=$(curl -s -H "Metadata-Flavor: Google" \
        http://metadata.google.internal/computeMetadata/v1/instance/attributes/nexus-url || echo "")
fi

if [[ -z "$NEXUS_URL" ]]; then
    echo "Error: NEXUS_URL is not set. Exiting."
    exit 1
fi

echo "Using Nexus URL: $NEXUS_URL"

# Configure Nexus Trusted Hosts
echo "Configuring trusted host..."
pip config set global.index-url "$NEXUS_URL"
pip config set global.trusted-host "$(echo $NEXUS_URL | awk -F/ '{print $3}')"

# Verify configuration
pip config list

echo "Post-startup script execution complete."


FROM gcr.io/deeplearning-platform-release/base-cpu

# Copy post-startup script
COPY post-startup.sh /usr/local/bin/post-startup.sh
RUN chmod +x /usr/local/bin/post-startup.sh

# Default Nexus URL (will be overridden dynamically)
ENV NEXUS_URL=""

# Ensure the script runs on startup
ENTRYPOINT ["/usr/local/bin/post-startup.sh"]

