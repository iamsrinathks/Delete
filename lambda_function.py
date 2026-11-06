"""
Copy Firehose-delivered Claude Code telemetry from S3 to Azure Blob Storage.

No stored secrets on either side:
  1. AWS STS GetWebIdentityToken (IAM outbound identity federation) issues a
     short-lived JWT for this function's execution role.
  2. Microsoft Entra ID accepts that JWT through a federated identity credential
     on a user-assigned managed identity and returns a Storage access token.
  3. The function lists recent S3 objects and uploads any that aren't already in
     the Blob container (same key, same size), so re-runs are safe.

Uses only the Lambda Python runtime (boto3 + standard library). Needs
botocore >= 1.41.0, the first release with sts.get_web_identity_token.

Environment variables (required):
  SOURCE_BUCKET            S3 bucket Firehose writes to
  AZURE_TENANT_ID          Entra tenant (directory) ID
  AZURE_CLIENT_ID          Client ID of the user-assigned managed identity
  AZURE_STORAGE_ACCOUNT    Storage account name
  AZURE_CONTAINER          Landing container name
Optional:
  SOURCE_PREFIX            Firehose S3 prefix, e.g. "claude-code/" (default "")
  LOOKBACK_DAYS            Days of YYYY/MM/DD/ folders to scan (default 2)
  AZURE_AUTHORITY_HOST     Default https://login.microsoftonline.com
  AZURE_BLOB_ENDPOINT      Default https://<account>.blob.core.windows.net
"""

import base64
import hashlib
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import formatdate

import boto3
import botocore

log = logging.getLogger()
log.setLevel(logging.INFO)

AUDIENCE = "api://AzureADTokenExchange"   # must match the federated credential
STORAGE_SCOPE = "https://storage.azure.com/.default"
BLOB_API_VERSION = "2023-11-03"           # Entra auth needs 2017-11-09 or later
MAX_SINGLE_PUT = 4_000 * 1024 * 1024      # Put Blob allows 5,000 MiB; stay under
SAFETY_MS = 60_000                        # stop with a minute of runtime left


def _env(name, default=None):
    value = os.environ.get(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing environment variable {name}")
    return value


def _jwt_claims(token):
    """Decode the JWT payload (no verification) so we can log iss/sub/aud."""
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def get_aws_web_identity_token(region):
    # GetWebIdentityToken is not available on the global STS endpoint.
    sts = boto3.client("sts", region_name=region,
                       endpoint_url=f"https://sts.{region}.amazonaws.com")
    resp = sts.get_web_identity_token(
        Audience=[AUDIENCE],
        SigningAlgorithm="RS256",   # Entra only accepts RS256-signed tokens
        DurationSeconds=300,
    )
    token = resp["WebIdentityToken"]
    claims = _jwt_claims(token)
    # These three values must match the Entra federated credential exactly.
    log.info("AWS token claims: iss=%s sub=%s aud=%s",
             claims.get("iss"), claims.get("sub"), claims.get("aud"))
    return token


def get_entra_storage_token(aws_jwt, tenant_id, client_id, authority):
    url = f"{authority.rstrip('/')}/{tenant_id}/oauth2/v2.0/token"
    body = urllib.parse.urlencode({
        "client_id": client_id,
        "scope": STORAGE_SCOPE,
        "grant_type": "client_credentials",
        "client_assertion_type":
            "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
        "client_assertion": aws_jwt,
    }).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["access_token"]
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        # AADSTS70021 = issuer/subject/audience mismatch, or the credential
        # was created moments ago and hasn't propagated yet.
        raise RuntimeError(f"Entra token request failed ({e.code}): {detail}")


def _blob_headers(token, extra=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "x-ms-version": BLOB_API_VERSION,
        "x-ms-date": formatdate(usegmt=True),
    }
    if extra:
        headers.update(extra)
    return headers


def blob_size(blob_url, token):
    """Return the blob's size, or None if it doesn't exist."""
    req = urllib.request.Request(blob_url, method="HEAD",
                                 headers=_blob_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return int(resp.headers["Content-Length"])
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def put_blob(blob_url, token, data):
    md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
    req = urllib.request.Request(blob_url, data=data, method="PUT",
                                 headers=_blob_headers(token, {
                                     "x-ms-blob-type": "BlockBlob",
                                     "Content-Length": str(len(data)),
                                     "Content-Type": "application/octet-stream",
                                     "Content-MD5": md5,  # Azure verifies it
                                 }))
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if resp.status != 201:
                raise RuntimeError(f"Put Blob returned {resp.status}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"Put Blob failed ({e.code}): {detail}")


def day_prefixes(base_prefix, days, now=None):
    now = now or datetime.now(timezone.utc)
    return [f"{base_prefix}{(now - timedelta(days=d)):%Y/%m/%d}/"
            for d in range(days)]


def lambda_handler(event, context):
    log.info("botocore %s", botocore.__version__)
    region = os.environ.get("AWS_REGION", "us-east-1")
    bucket = _env("SOURCE_BUCKET")
    base_prefix = os.environ.get("SOURCE_PREFIX", "")
    days = int(os.environ.get("LOOKBACK_DAYS", "2"))
    account = _env("AZURE_STORAGE_ACCOUNT")
    container = _env("AZURE_CONTAINER")
    blob_endpoint = os.environ.get(
        "AZURE_BLOB_ENDPOINT", f"https://{account}.blob.core.windows.net")
    authority = os.environ.get("AZURE_AUTHORITY_HOST",
                               "https://login.microsoftonline.com")

    aws_jwt = get_aws_web_identity_token(region)
    token = get_entra_storage_token(aws_jwt, _env("AZURE_TENANT_ID"),
                                    _env("AZURE_CLIENT_ID"), authority)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    copied = skipped = copied_bytes = 0
    failures = []
    stopped_early = False

    for prefix in day_prefixes(base_prefix, days):
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if context and context.get_remaining_time_in_millis() < SAFETY_MS:
                    stopped_early = True
                    break
                key, size = obj["Key"], obj["Size"]
                if key.endswith("/"):
                    continue
                blob_url = (f"{blob_endpoint.rstrip('/')}/{container}/"
                            f"{urllib.parse.quote(key)}")
                try:
                    if blob_size(blob_url, token) == size:
                        skipped += 1
                        continue
                    if size > MAX_SINGLE_PUT:
                        raise RuntimeError(f"{key} is {size} bytes; too big "
                                           "for a single Put Blob")
                    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                    put_blob(blob_url, token, data)
                    copied += 1
                    copied_bytes += size
                    log.info("copied s3://%s/%s (%d bytes)", bucket, key, size)
                except Exception as e:  # keep going, report at the end
                    log.error("failed %s: %s", key, e)
                    failures.append(key)
            if stopped_early:
                break
        if stopped_early:
            break

    result = {"copied": copied, "skipped_existing": skipped,
              "copied_bytes": copied_bytes, "failed": len(failures),
              "stopped_early_for_timeout": stopped_early}
    log.info("result %s", json.dumps(result))
    if failures:
        # Fail the invocation so the error shows up in metrics and alarms.
        # Files already copied are skipped on the next run.
        raise RuntimeError(f"{len(failures)} object(s) failed: {failures[:10]}")
    return result
