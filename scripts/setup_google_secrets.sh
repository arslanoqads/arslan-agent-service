#!/usr/bin/env bash
# One-time Secret Manager setup. Do not commit secret values.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-project-a4383faa-cfc2-4119-8d5}"

put_secret() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    echo "Skip ${name}: empty"
    return
  fi
  if gcloud secrets describe "${name}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    printf '%s' "${value}" | gcloud secrets versions add "${name}" --project="${PROJECT_ID}" --data-file=-
  else
    printf '%s' "${value}" | gcloud secrets create "${name}" --project="${PROJECT_ID}" --data-file=-
  fi
  echo "Stored ${name}"
}

put_secret GOOGLE_OAUTH_CLIENT_ID "${GOOGLE_OAUTH_CLIENT_ID:-}"
put_secret GOOGLE_OAUTH_CLIENT_SECRET "${GOOGLE_OAUTH_CLIENT_SECRET:-}"
put_secret GOOGLE_REFRESH_TOKEN "${GOOGLE_REFRESH_TOKEN:-}"
put_secret GOOGLE_SENDER_EMAIL "${GOOGLE_SENDER_EMAIL:-thatqadri@gmail.com}"
put_secret PUBLIC_WEBSITE_URL "${PUBLIC_WEBSITE_URL:-https://qads.us}"
put_secret PUBLIC_LINKEDIN_URL "${PUBLIC_LINKEDIN_URL:-https://www.linkedin.com/in/maqadri}"
put_secret PUBLIC_INSTAGRAM_URL "${PUBLIC_INSTAGRAM_URL:-}"
put_secret PUBLIC_SUBSTACK_URL "${PUBLIC_SUBSTACK_URL:-}"
put_secret OBSERVABILITY_TOKEN "${OBSERVABILITY_TOKEN:-}"
