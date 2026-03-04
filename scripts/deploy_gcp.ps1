# GCP Deployment Script for YGN-SAGE MCP Gateway
# Ensure you are authenticated: gcloud auth login

$PROJECT_ID = gcloud config get-value project
if (-not $PROJECT_ID) {
    Write-Host "Erreur : Aucun projet GCP sélectionné. Veuillez exécuter 'gcloud init' ou 'gcloud config set project VOTRE_PROJET'." -ForegroundColor Red
    exit 1
}

$SERVICE_NAME = "ygn-sage-mcp"
$REGION = "europe-west1" # Optimal pour minimiser la latence depuis la France

Write-Host "🚀 Déploiement de l'architecture YGN-SAGE sur Google Cloud Run (Projet: $PROJECT_ID)..." -ForegroundColor Cyan

# Soumission du build et déploiement via Cloud Build & Cloud Run
gcloud run deploy $SERVICE_NAME `
    --source . `
    --port 8080 `
    --region $REGION `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 10

Write-Host "✅ Déploiement terminé. L'API (MCP Gateway) est désormais scalable et publique." -ForegroundColor Green
