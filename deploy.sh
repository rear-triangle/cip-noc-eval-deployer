#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="lmic-dev-datahub"
REGION="us-east4"
REPO="cip-noc-eval-deployer"
IMAGE="cip-noc-eval-deployer"
JOB="cip-noc-eval"

OLLAMA_URL="https://ollama-gcs-llama3-gpu-228291553312.us-east4.run.app"

PROMPT_VERSION="v010"
PROMPT_PATH="/app/prompts/v010_prompt.txt"
CONFIG_PATH="/app/configs/dev.yaml"

MAX_PAIRS=1000
CIP_LEVELS=3
NOC_LEVELS=5

TAG="${PROMPT_VERSION}-$(date -u +%Y%m%dT%H%M%SZ)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "JOB=${JOB}"
echo "PROMPT_VERSION=${PROMPT_VERSION}"
echo "PROMPT_PATH=${PROMPT_PATH}"
echo "TAG=${TAG}"
echo "IMAGE_URI=${IMAGE_URI}"

# Basic checks (fail early with useful messages)
command -v gcloud >/dev/null || { echo "gcloud not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "docker buildx not available"; exit 1; }

# Auth for Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build + push image
docker buildx build \
  --platform linux/amd64 \
  -t "${IMAGE_URI}" \
  --push \
  .

echo "Pushed: ${IMAGE_URI}"

# Ensure job exists (readable error if not)
gcloud run jobs describe "${JOB}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --quiet >/dev/null

# Update job image
gcloud run jobs update "${JOB}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE_URI}" \
  --quiet

echo "Job image now:"
gcloud run jobs describe "${JOB}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format="value(template.template.containers[0].image)"

RUN_ID="run_${MAX_PAIRS}pair_${PROMPT_VERSION}_$(date -u +%Y%m%dT%H%M%SZ)"

# NOTE: Avoid commas in RUN_NOTES because --update-env-vars uses commas as separators.
RUN_NOTES="prompt=${PROMPT_VERSION}; pairs=${MAX_PAIRS}; cip=${CIP_LEVELS}; noc=${NOC_LEVELS}; ollama_url=${OLLAMA_URL}; img=${TAG}; prompt_path=${PROMPT_PATH}"

echo "RUN_ID=${RUN_ID}"
echo "RUN_NOTES=${RUN_NOTES}"

gcloud run jobs execute "${JOB}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --quiet \
  --update-env-vars \
"CONFIG_PATH=${CONFIG_PATH},RUN_ID=${RUN_ID},RUN_NOTES=${RUN_NOTES},PROMPT_PATH=${PROMPT_PATH},PROMPT_VERSION=${PROMPT_VERSION},CIP_LEVELS=${CIP_LEVELS},NOC_LEVELS=${NOC_LEVELS},MAX_PAIRS=${MAX_PAIRS},OLLAMA_URL=${OLLAMA_URL}"