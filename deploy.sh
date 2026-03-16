#!/bin/bash
# ── deploy.sh — Litterman AI Dashboard → Cloud Run ───────────────────────────
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Requirements:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - GEMINI_API_KEY stored in Google Secret Manager (one-time setup):
#       gcloud secrets create GEMINI_API_KEY --data-file=- <<< "YOUR_KEY"
#       gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
#         --member "serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
#         --role "roles/secretmanager.secretAccessor"
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-litterman-ai}"
REGION="us-central1"
SERVICE_NAME="litterman-dashboard"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

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
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest" \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 1 \
    --max-instances 1 \
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