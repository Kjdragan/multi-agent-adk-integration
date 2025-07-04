#!/bin/bash

# Multi-Agent Research Platform - Cloud Run Deployment Script
# This script automates the deployment process to Google Cloud Run

set -euo pipefail  # Exit on any error, undefined variable, or pipe failure

# Configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_CONFIG="$SCRIPT_DIR/config"

# Default values
DEFAULT_PROJECT_ID=""
DEFAULT_REGION="us-central1"
DEFAULT_SERVICE_NAME="multi-agent-platform"
DEFAULT_IMAGE_NAME="gcr.io/\${PROJECT_ID}/multi-agent-platform"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Multi-Agent Research Platform - Cloud Run Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -p, --project-id PROJECT_ID     Google Cloud Project ID
    -r, --region REGION             Deployment region (default: us-central1)
    -s, --service-name NAME         Cloud Run service name (default: multi-agent-platform)
    -e, --environment ENV           Environment (staging|production) (default: production)
    -b, --branch BRANCH             Git branch to deploy (default: current branch)
    -t, --tag TAG                   Docker image tag (default: latest)
    -i, --image IMAGE               Full image name (overrides default)
    -f, --force                     Force deployment without confirmation
    -d, --dry-run                   Show what would be done without executing
    -v, --verbose                   Enable verbose output
    -h, --help                      Show this help message

EXAMPLES:
    # Deploy to production
    $0 --project-id my-project --environment production

    # Deploy staging with custom service name
    $0 -p my-project -e staging -s my-service-staging

    # Dry run to see what would happen
    $0 -p my-project -d

    # Deploy specific branch with verbose output
    $0 -p my-project -b feature/new-agent -v

PREREQUISITES:
    1. Google Cloud SDK installed and authenticated
    2. Docker installed
    3. Required APIs enabled: Cloud Run, Cloud Build, Container Registry
    4. Proper IAM permissions configured

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--project-id)
                PROJECT_ID="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -s|--service-name)
                SERVICE_NAME="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -b|--branch)
                BRANCH="$2"
                shift 2
                ;;
            -t|--tag)
                TAG="$2"
                shift 2
                ;;
            -i|--image)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Initialize variables with defaults
init_variables() {
    PROJECT_ID=${PROJECT_ID:-$DEFAULT_PROJECT_ID}
    REGION=${REGION:-$DEFAULT_REGION}
    SERVICE_NAME=${SERVICE_NAME:-$DEFAULT_SERVICE_NAME}
    ENVIRONMENT=${ENVIRONMENT:-production}
    BRANCH=${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}
    TAG=${TAG:-latest}
    IMAGE_NAME=${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}
    FORCE=${FORCE:-false}
    DRY_RUN=${DRY_RUN:-false}
    VERBOSE=${VERBOSE:-false}
    
    # Replace placeholder in image name
    IMAGE_NAME=${IMAGE_NAME//\$\{PROJECT_ID\}/$PROJECT_ID}
    
    # Set verbose mode for gcloud if requested
    if [[ "$VERBOSE" == "true" ]]; then
        GCLOUD_VERBOSITY="--verbosity=debug"
    else
        GCLOUD_VERBOSITY="--verbosity=info"
    fi
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud SDK is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if authenticated with gcloud
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with Google Cloud SDK. Run 'gcloud auth login'"
        exit 1
    fi
    
    # Validate project ID
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required. Use --project-id or set DEFAULT_PROJECT_ID"
        exit 1
    fi
    
    # Check if project exists and is accessible
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_error "Project '$PROJECT_ID' not found or not accessible"
        exit 1
    fi
    
    # Validate environment
    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Environment must be 'staging' or 'production'"
        exit 1
    fi
    
    log_success "Prerequisites validated"
}

# Setup Google Cloud project
setup_project() {
    log_info "Setting up Google Cloud project..."
    
    # Set active project
    gcloud config set project "$PROJECT_ID" $GCLOUD_VERBOSITY
    
    # Enable required APIs
    log_info "Enabling required APIs..."
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        containerregistry.googleapis.com \
        $GCLOUD_VERBOSITY
    
    log_success "Project setup completed"
}

# Build and push Docker image
build_image() {
    log_info "Building Docker image..."
    
    local dockerfile_path="$SCRIPT_DIR/Dockerfile"
    local image_tag="$IMAGE_NAME:$TAG"
    local build_context="$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build image: $image_tag"
        log_info "[DRY RUN] Using Dockerfile: $dockerfile_path"
        log_info "[DRY RUN] Build context: $build_context"
        return 0
    fi
    
    # Build image using Cloud Build for better performance and caching
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Building with Cloud Build..."
        gcloud builds submit \
            --tag="$image_tag" \
            --dockerfile="$dockerfile_path" \
            "$build_context" \
            $GCLOUD_VERBOSITY
    else
        gcloud builds submit \
            --tag="$image_tag" \
            --dockerfile="$dockerfile_path" \
            "$build_context" \
            --quiet
    fi
    
    log_success "Image built and pushed: $image_tag"
}

# Deploy to Cloud Run
deploy_service() {
    log_info "Deploying to Cloud Run..."
    
    local image_tag="$IMAGE_NAME:$TAG"
    local deploy_service_name="$SERVICE_NAME"
    
    # Adjust service name for staging
    if [[ "$ENVIRONMENT" == "staging" ]]; then
        deploy_service_name="$SERVICE_NAME-staging"
    fi
    
    # Prepare deployment command
    local deploy_cmd=(
        gcloud run deploy "$deploy_service_name"
        --image="$image_tag"
        --region="$REGION"
        --platform=managed
        --allow-unauthenticated
    )
    
    # Environment-specific configuration
    if [[ "$ENVIRONMENT" == "production" ]]; then
        deploy_cmd+=(
            --memory=4Gi
            --cpu=4
            --timeout=600
            --max-instances=50
            --min-instances=2
            --concurrency=10
        )
    else
        deploy_cmd+=(
            --memory=2Gi
            --cpu=2
            --timeout=300
            --max-instances=10
            --min-instances=1
            --concurrency=5
        )
    fi
    
    # Environment variables
    deploy_cmd+=(
        --set-env-vars="ENVIRONMENT=$ENVIRONMENT,GOOGLE_GENAI_USE_VERTEXAI=true"
    )
    
    # Add verbosity
    deploy_cmd+=("$GCLOUD_VERBOSITY")
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: ${deploy_cmd[*]}"
        return 0
    fi
    
    # Execute deployment
    "${deploy_cmd[@]}"
    
    # Get service URL
    local service_url
    service_url=$(gcloud run services describe "$deploy_service_name" \
        --region="$REGION" \
        --format="value(status.url)")
    
    log_success "Service deployed successfully!"
    log_success "Service URL: $service_url"
}

# Run health check
health_check() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform health check"
        return 0
    fi
    
    log_info "Performing health check..."
    
    local deploy_service_name="$SERVICE_NAME"
    if [[ "$ENVIRONMENT" == "staging" ]]; then
        deploy_service_name="$SERVICE_NAME-staging"
    fi
    
    local service_url
    service_url=$(gcloud run services describe "$deploy_service_name" \
        --region="$REGION" \
        --format="value(status.url)")
    
    # Wait for service to be ready
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$service_url/health" > /dev/null; then
            log_success "Health check passed!"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Display deployment summary
show_summary() {
    log_info "Deployment Summary:"
    echo "  Project ID: $PROJECT_ID"
    echo "  Region: $REGION"
    echo "  Service Name: $SERVICE_NAME"
    echo "  Environment: $ENVIRONMENT"
    echo "  Branch: $BRANCH"
    echo "  Image: $IMAGE_NAME:$TAG"
    echo "  Dry Run: $DRY_RUN"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        local deploy_service_name="$SERVICE_NAME"
        if [[ "$ENVIRONMENT" == "staging" ]]; then
            deploy_service_name="$SERVICE_NAME-staging"
        fi
        
        local service_url
        service_url=$(gcloud run services describe "$deploy_service_name" \
            --region="$REGION" \
            --format="value(status.url)" 2>/dev/null || echo "N/A")
        
        echo "  Service URL: $service_url"
    fi
}

# Confirmation prompt
confirm_deployment() {
    if [[ "$FORCE" == "true" || "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    
    echo
    show_summary
    echo
    read -p "Proceed with deployment? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi
}

# Main execution function
main() {
    log_info "Multi-Agent Research Platform - Cloud Run Deployment"
    echo
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    parse_arguments "$@"
    init_variables
    validate_prerequisites
    confirm_deployment
    
    echo
    log_info "Starting deployment process..."
    
    setup_project
    build_image
    deploy_service
    health_check
    
    echo
    log_success "Deployment completed successfully!"
    show_summary
}

# Run main function with all arguments
main "$@"