#!/bin/bash

# Multi-Agent Research Platform - Cloud Run Setup Script
# This script sets up the necessary Google Cloud resources for deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
Multi-Agent Research Platform - Cloud Run Setup Script

This script sets up the necessary Google Cloud resources for deploying
the Multi-Agent Research Platform to Cloud Run.

Usage: $0 [OPTIONS]

OPTIONS:
    -p, --project-id PROJECT_ID     Google Cloud Project ID (required)
    -r, --region REGION             Deployment region (default: us-central1)
    -s, --service-account NAME      Service account name (default: multi-agent-platform)
    -b, --billing-account ID        Billing account ID (for new projects)
    --create-project                Create new project if it doesn't exist
    --skip-billing                  Skip billing account setup
    --dry-run                       Show what would be done without executing
    -v, --verbose                   Enable verbose output
    -h, --help                      Show this help message

EXAMPLES:
    # Setup existing project
    $0 --project-id my-existing-project

    # Create new project with billing
    $0 --project-id my-new-project --create-project --billing-account 123456-789ABC-DEF012

    # Setup with custom region and service account
    $0 -p my-project -r europe-west1 -s my-service-account

PREREQUISITES:
    1. Google Cloud SDK installed and authenticated
    2. Appropriate permissions to create resources
    3. Billing account (for new projects)

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
            -s|--service-account)
                SERVICE_ACCOUNT_NAME="$2"
                shift 2
                ;;
            -b|--billing-account)
                BILLING_ACCOUNT="$2"
                shift 2
                ;;
            --create-project)
                CREATE_PROJECT=true
                shift
                ;;
            --skip-billing)
                SKIP_BILLING=true
                shift
                ;;
            --dry-run)
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

# Initialize variables
init_variables() {
    PROJECT_ID=${PROJECT_ID:-}
    REGION=${REGION:-us-central1}
    SERVICE_ACCOUNT_NAME=${SERVICE_ACCOUNT_NAME:-multi-agent-platform}
    BILLING_ACCOUNT=${BILLING_ACCOUNT:-}
    CREATE_PROJECT=${CREATE_PROJECT:-false}
    SKIP_BILLING=${SKIP_BILLING:-false}
    DRY_RUN=${DRY_RUN:-false}
    VERBOSE=${VERBOSE:-false}
    
    # Derived variables
    SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
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
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Not authenticated with Google Cloud SDK. Run 'gcloud auth login'"
        exit 1
    fi
    
    # Validate project ID
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "Project ID is required. Use --project-id"
        exit 1
    fi
    
    # Validate billing account if creating project
    if [[ "$CREATE_PROJECT" == "true" && "$SKIP_BILLING" == "false" && -z "$BILLING_ACCOUNT" ]]; then
        log_error "Billing account is required when creating a new project. Use --billing-account or --skip-billing"
        exit 1
    fi
    
    log_success "Prerequisites validated"
}

# Create or validate project
setup_project() {
    log_info "Setting up project: $PROJECT_ID"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would setup project: $PROJECT_ID"
        return 0
    fi
    
    # Check if project exists
    if gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_info "Project $PROJECT_ID already exists"
    elif [[ "$CREATE_PROJECT" == "true" ]]; then
        log_info "Creating project: $PROJECT_ID"
        gcloud projects create "$PROJECT_ID" $GCLOUD_VERBOSITY
        
        # Link billing account
        if [[ "$SKIP_BILLING" == "false" ]]; then
            log_info "Linking billing account: $BILLING_ACCOUNT"
            gcloud billing projects link "$PROJECT_ID" \
                --billing-account="$BILLING_ACCOUNT" \
                $GCLOUD_VERBOSITY
        fi
    else
        log_error "Project $PROJECT_ID does not exist. Use --create-project to create it"
        exit 1
    fi
    
    # Set active project
    gcloud config set project "$PROJECT_ID" $GCLOUD_VERBOSITY
    
    log_success "Project setup completed"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required APIs..."
    
    local apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "containerregistry.googleapis.com"
        "secretmanager.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "clouddebugger.googleapis.com"
        "cloudtrace.googleapis.com"
        "cloudprofiler.googleapis.com"
    )
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would enable APIs: ${apis[*]}"
        return 0
    fi
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" $GCLOUD_VERBOSITY
    done
    
    log_success "APIs enabled successfully"
}

# Create service account
create_service_account() {
    log_info "Creating service account: $SERVICE_ACCOUNT_NAME"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create service account: $SERVICE_ACCOUNT_EMAIL"
        return 0
    fi
    
    # Check if service account exists
    if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &> /dev/null; then
        log_info "Service account already exists: $SERVICE_ACCOUNT_EMAIL"
    else
        # Create service account
        gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
            --display-name="Multi-Agent Platform Service Account" \
            --description="Service account for Multi-Agent Research Platform Cloud Run deployment" \
            $GCLOUD_VERBOSITY
    fi
    
    # Grant necessary roles
    local roles=(
        "roles/run.invoker"
        "roles/cloudsql.client"
        "roles/secretmanager.secretAccessor"
        "roles/aiplatform.user"
        "roles/logging.logWriter"
        "roles/monitoring.metricWriter"
        "roles/cloudtrace.agent"
    )
    
    for role in "${roles[@]}"; do
        log_info "Granting role $role to service account..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
            --role="$role" \
            $GCLOUD_VERBOSITY > /dev/null
    done
    
    log_success "Service account created and configured"
}

# Create secrets
create_secrets() {
    log_info "Creating secrets in Secret Manager..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create secrets in Secret Manager"
        return 0
    fi
    
    # Create placeholder secrets (users will need to update these)
    local secrets=(
        "google-api-key"
        "openweather-api-key"
        "perplexity-api-key"
        "tavily-api-key"
    )
    
    for secret in "${secrets[@]}"; do
        if ! gcloud secrets describe "$secret" &> /dev/null; then
            log_info "Creating secret: $secret"
            echo "PLACEHOLDER_VALUE" | gcloud secrets create "$secret" \
                --data-file=- \
                $GCLOUD_VERBOSITY
            
            # Grant service account access
            gcloud secrets add-iam-policy-binding "$secret" \
                --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
                --role="roles/secretmanager.secretAccessor" \
                $GCLOUD_VERBOSITY > /dev/null
        else
            log_info "Secret already exists: $secret"
        fi
    done
    
    log_warning "Remember to update the secret values:"
    for secret in "${secrets[@]}"; do
        echo "  gcloud secrets versions add $secret --data-file=<(echo 'YOUR_ACTUAL_SECRET')"
    done
    
    log_success "Secrets created"
}

# Setup Cloud Build triggers
setup_build_triggers() {
    log_info "Setting up Cloud Build triggers..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would setup Cloud Build triggers"
        return 0
    fi
    
    # Grant Cloud Build service account necessary permissions
    local cloudbuild_sa
    cloudbuild_sa=$(gcloud projects get-iam-policy "$PROJECT_ID" \
        --flatten="bindings[].members" \
        --format="table(bindings.members)" \
        --filter="bindings.role:roles/cloudbuild.builds.builder" | \
        grep "@cloudbuild.gserviceaccount.com" | head -1)
    
    if [[ -n "$cloudbuild_sa" ]]; then
        local roles=(
            "roles/run.admin"
            "roles/iam.serviceAccountUser"
            "roles/storage.admin"
        )
        
        for role in "${roles[@]}"; do
            gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="$cloudbuild_sa" \
                --role="$role" \
                $GCLOUD_VERBOSITY > /dev/null
        done
    fi
    
    log_success "Cloud Build permissions configured"
}

# Create VPC connector (optional)
create_vpc_connector() {
    log_info "Creating VPC connector for private networking..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create VPC connector"
        return 0
    fi
    
    # Check if VPC connector already exists
    if ! gcloud compute networks vpc-access connectors describe default-connector \
        --region="$REGION" &> /dev/null; then
        
        log_info "Creating VPC connector..."
        gcloud compute networks vpc-access connectors create default-connector \
            --region="$REGION" \
            --subnet-project="$PROJECT_ID" \
            --range="10.8.0.0/28" \
            $GCLOUD_VERBOSITY
    else
        log_info "VPC connector already exists"
    fi
    
    log_success "VPC connector configured"
}

# Generate configuration files
generate_configs() {
    log_info "Generating configuration files..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would generate configuration files"
        return 0
    fi
    
    # Generate .env file for deployment
    cat > .env.deployment << EOF
# Generated Cloud Run deployment configuration
PROJECT_ID=$PROJECT_ID
REGION=$REGION
SERVICE_ACCOUNT_EMAIL=$SERVICE_ACCOUNT_EMAIL
SERVICE_NAME=multi-agent-platform

# Image configuration
IMAGE_NAME=gcr.io/\${PROJECT_ID}/multi-agent-platform

# Deployment settings
STAGING_MEMORY=2Gi
STAGING_CPU=2
PRODUCTION_MEMORY=4Gi
PRODUCTION_CPU=4

# Generated on: $(date)
EOF
    
    # Update service.yaml with project-specific values
    sed -i "s/PROJECT_ID/$PROJECT_ID/g" deployment/cloud-run/service.yaml
    sed -i "s/REGION/$REGION/g" deployment/cloud-run/service.yaml
    
    log_success "Configuration files generated"
}

# Display setup summary
show_summary() {
    echo
    log_success "Cloud Run setup completed successfully!"
    echo
    echo "Project Configuration:"
    echo "  Project ID: $PROJECT_ID"
    echo "  Region: $REGION"
    echo "  Service Account: $SERVICE_ACCOUNT_EMAIL"
    echo "  VPC Connector: default-connector"
    echo
    echo "Next Steps:"
    echo "1. Update secret values with your actual API keys:"
    echo "   gcloud secrets versions add google-api-key --data-file=<(echo 'YOUR_GOOGLE_API_KEY')"
    echo "   gcloud secrets versions add openweather-api-key --data-file=<(echo 'YOUR_OPENWEATHER_KEY')"
    echo
    echo "2. Deploy using the deployment script:"
    echo "   ./deployment/cloud-run/deploy.sh --project-id $PROJECT_ID"
    echo
    echo "3. Or use GitHub Actions by setting these repository secrets:"
    echo "   - GCP_PROJECT_ID: $PROJECT_ID"
    echo "   - GCP_SA_KEY: <service-account-json-key>"
    echo "   - GOOGLE_API_KEY: <your-google-api-key>"
    echo "   - OPENWEATHER_API_KEY: <your-openweather-key>"
    echo
}

# Main execution
main() {
    log_info "Multi-Agent Research Platform - Cloud Run Setup"
    echo
    
    parse_arguments "$@"
    init_variables
    validate_prerequisites
    
    if [[ "$DRY_RUN" == "false" ]]; then
        echo "Setup Configuration:"
        echo "  Project ID: $PROJECT_ID"
        echo "  Region: $REGION"
        echo "  Service Account: $SERVICE_ACCOUNT_NAME"
        echo "  Create Project: $CREATE_PROJECT"
        echo
        read -p "Continue with setup? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Setup cancelled"
            exit 0
        fi
    fi
    
    echo
    log_info "Starting Cloud Run setup..."
    
    setup_project
    enable_apis
    create_service_account
    create_secrets
    setup_build_triggers
    create_vpc_connector
    generate_configs
    
    show_summary
}

# Run main function
main "$@"