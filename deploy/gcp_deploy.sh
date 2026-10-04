#!/usr/bin/env bash
# ==============================================================================
# GCP Cloud Run Deployment Automation Script for Multilingual CS Agent
# ==============================================================================
set -e

PROJECT_ID=${GCP_PROJECT_ID:-"your-gcp-project-id"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="multilingual-cs-agent"
REPO_NAME="cs-agent-repo"
IMAGE_TAG="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=== 1. Setting Active GCP Project ==="
gcloud config set project "${PROJECT_ID}"

echo "=== 2. Enabling Required GCP APIs ==="
gcloud services enable \
    run.googleapis.com \
    containerregistry.googleapis.com \
    cloudbuild.googleapis.com

echo "=== 3. Building and Pushing Container via Cloud Build ==="
gcloud builds submit --tag "${IMAGE_TAG}" .

echo "=== 4. Deploying to Cloud Run (Serverless) ==="
# Cloud Run configurations tuned for sub-3s SLA:
# - CPU: 2 vCPU
# - Memory: 4Gi (sufficient for FAISS in-memory index & embeddings)
# - Concurrency: 80 requests/container
# - Min instances: 1 (eliminates cold starts to preserve <3s SLA)
gcloud run deploy "${SERVICE_NAME}" \
    --image "${IMAGE_TAG}" \
    --platform managed \
    --region "${REGION}" \
    --allow-unauthenticated \
    --port 8080 \
    --cpu 2 \
    --memory 4Gi \
    --min-instances 1 \
    --max-instances 10 \
    --concurrency 80 \
    --timeout 300s

echo "=== 5. Deployment Completed ==="
gcloud run services describe "${SERVICE_NAME}" --platform managed --region "${REGION}" --format="value(status.url)"
