#!/bin/bash

# Enhanced rebuild script with better error handling, logging, and modularity
set -euo pipefail  # More strict error handling
exec 1> >(tee -a rebuild.log) 2>&1  # Log all output

# Configuration
ENVIRONMENT=${ENVIRONMENT:-development}
BUILD_APP=true
BUILD_DOCS=true
BUILD_JUPYTER=true
SKIP_CLEANUP=false
export COMPOSE_BAKE=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-app) BUILD_APP=false; shift ;;
        --skip-docs) BUILD_DOCS=false; shift ;;
        --skip-jupyter) BUILD_JUPYTER=false; shift ;;
        --skip-cleanup) SKIP_CLEANUP=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Start time tracking
start_time=$(date +%s)

echo -e "${BLUE}🚀 Starting rebuild process for environment: $ENVIRONMENT${NC}"
echo "Timestamp: $(date)"

# Safety checks
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
        exit 1
    fi

    # Check if required files exist
    if [[ ! -f "docker-compose.yaml" ]]; then
        echo -e "${RED}❌ docker-compose.yaml not found in current directory.${NC}"
        exit 1
    fi

    if [[ ! -f "docker/pyproject.toml" ]]; then
        echo -e "${RED}❌ docker/pyproject.toml not found.${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Cleanup function
cleanup() {
    echo -e "${BLUE}🧹 Performing cleanup..."

    if [[ "${SKIP_CLEANUP:-false}" == "true" ]]; then
        echo -e "${YELLOW}⚠️ Skipping cleanup as requested${NC}"
        return
    fi

    # Clean Python cache files
    echo -e "${BLUE}🧹 Cleaning Python cache files..."
    sudo find . | grep -E "(\.ipynb_checkpoints|__pycache__|\.pyc|\.pyo$|\.Trash-0)" | xargs sudo rm -rf 2>/dev/null || true

    # Fix file ownership
    echo -e "${BLUE}🔒 Fixing file ownership...${NC}"
    sudo chown -R ${USER}:${USER} .

    # Clean chatbot storage
    echo -e "${BLUE}🤖 Cleaning chatbot storage...${NC}"
    rm -rf tmp/chatbot_storage/*

    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Docker cleanup function
docker_cleanup() {
    echo -e "${BLUE}🐳 Cleaning up Docker environment...${NC}"

    docker compose down --remove-orphans > /dev/null 2>&1
    docker system prune -f > /dev/null 2>&1

    echo -e "${GREEN}✅ Docker cleanup completed${NC}"
}

# UV build function
build_with_uv() {
    local extra_deps="$1"
    local target="$2"
    local service_name="$3"

    echo -e "${BLUE}🔨 Building $service_name with UV...${NC}"

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        echo "DRY RUN: Would run UV sync for $service_name with extra deps: $extra_deps"
        echo "DRY RUN: Would build Docker target: $target"
        return
    fi

    docker run -it --rm -v ./docker:/docker --workdir /docker --entrypoint uv \
        ghcr.io/astral-sh/uv:python3.13-bookworm-slim \
        sync --no-progress --quiet --link-mode=copy --compile-bytecode ${extra_deps} || { echo -e "${RED}❌ UV sync failed for $service_name${NC}"; exit 1; }

    COMPOSE_BAKE=true docker compose build $target || { echo -e "${RED}❌ Docker build failed for $service_name${NC}"; exit 1; }

    echo -e "${GREEN}✅ $service_name build completed${NC}"
}

# Health check function
health_check() {
    echo -e "${BLUE}🏥 Performing health checks...${NC}"

    # Wait for services to be ready
    echo -e "${BLUE}🕛 Waiting for services to be ready...${NC}"
    sleep 10

    # Check app service
    if curl -f http://localhost:8501/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ App service is running${NC}"
    else
        echo -e "${RED}❌ App service failed to start${NC}"
        return 1
    fi

    # Check docs service
    if curl -f http://localhost:8001/readme/ > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Docs service is running${NC}"
    else
        echo -e "${RED}❌ Docs service failed to start${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ All health checks passed${NC}"
}

# Main execution
main() {
    clear
    echo -e "${BLUE}🎯 Starting rebuild process...${NC}"

    # Check prerequisites
    check_prerequisites

    # Cleanup
    cleanup

    # Docker cleanup
    docker_cleanup

    # Build services
    if [[ "$BUILD_APP" == "true" ]]; then
        build_with_uv "" "app" "App"
    fi

    if [[ "$BUILD_DOCS" == "true" ]]; then
        build_with_uv "--extra streamlit-docs" "docs" "Docs"
    fi

    if [[ "$BUILD_JUPYTER" == "true" ]]; then
        build_with_uv "--extra jupyterlab" "jupyter" "Jupyter"
    fi

    # Start services
    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        echo -e "${BLUE}🚀 Starting services...${NC}"
        docker compose up app docs nginx -d || { echo -e "${RED}❌ Failed to start services${NC}"; exit 1; }

        # echo -e "${BLUE}📋 Service status:${NC}"
        # docker compose ps

        # Health checks
        if health_check; then
            echo -e "${GREEN}🎉 Rebuild completed successfully!${NC}"
        else
            echo -e "${RED}❌ Some services failed health checks${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}DRY RUN: Would start services with 'docker compose up app docs nginx -d'${NC}"
    fi

    # Calculate and display timing
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo -e "${GREEN}✅ Rebuild completed in ${duration} seconds${NC}"

    if [[ "${DRY_RUN:-false}" != "true" ]]; then
        echo -e "${BLUE}📋 Following app logs (Ctrl+C to stop):${NC}"
        docker compose logs -f app
    fi
}

# Run main function
main "$@"
