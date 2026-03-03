#!/bin/bash
# ── deploy.sh — Litterman AI Dashboard → Cloud Run ───────────────────────────
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Requirements:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Docker installed (or gcloud builds submit used instead)
#   - PROJECT_ID set below or exported as env var before running
#
# This script is included in the repository as proof of automated deployment
# for the Gemini Live Agent Challenge bonus criteria.
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-litterman-ai}"
REGION="us-central1"
SERVICE_NAME="litterman-dashboard"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
SERVICE_ACCOUNT="${SERVICE_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "================================================"
echo " Litterman AI — Cloud Run Deploy"
echo " Project : ${PROJECT_ID}"
echo " Region  : ${REGION}"
echo " Service : ${SERVICE_NAME}"
echo "================================================"

# ── Step 1: Set active project ────────────────────────────────────────────────
echo ""
echo "[1/4] Setting active GCP project..."
gcloud config set project "${PROJECT_ID}"

# ── Step 2: Build and push image via Cloud Build ──────────────────────────────
# Uses Cloud Build instead of local Docker — no Docker daemon required locally.
echo ""
echo "[2/4] Building and pushing image via Cloud Build..."
gcloud builds submit \
    --tag "${IMAGE}" \
    --project "${PROJECT_ID}"

# ── Step 3: Deploy to Cloud Run ───────────────────────────────────────────────
echo ""
echo "[3/4] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE}" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --service-account "${SERVICE_ACCOUNT}" \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --port 8080 \
    --project "${PROJECT_ID}"

# ── Step 4: Print service URL ─────────────────────────────────────────────────
echo ""
echo "[4/4] Deploy complete. Fetching service URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --region "${REGION}" \
    --project "${PROJECT_ID}" \
    --format "value(status.url)")

echo ""
echo "================================================"
echo " ✓ Dashboard live at:"
echo "   ${SERVICE_URL}"
echo "================================================"