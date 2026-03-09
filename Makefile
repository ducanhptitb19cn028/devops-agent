# ── DevOps AI Agent — Makefile ────────────────────────────────
# Convenient shortcuts for building, deploying, and managing
# the TraceFlix observability agent.
#
# Usage:
#   make build         — Build all Docker images
#   make deploy        — Deploy to K8s (Claude mode)
#   make deploy-ml     — Deploy with ML model server
#   make train         — Train ML models
#   make test-ml       — Run ML smoke tests
#   make status        — Show pod status
#   make dashboard     — Port-forward dashboard
#   make clean         — Tear down all resources

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Configuration ────────────────────────────────────────────
NAMESPACE := devops-agent
MINIKUBE := $(shell where minikube 2>NUL)

# Load credentials from .env (gitignored) — create .env from .env.example first run
-include .env
export

# ── Help ─────────────────────────────────────────────────────
.PHONY: help
help: ## Show available commands
	@echo "DevOps AI Agent Make Targets"

# ── Build ────────────────────────────────────────────────────
.PHONY: build build-minikube build-ml

load-infra-images: ## Pull and load infra images into Docker Desktop k8s node
	docker pull redis:7.4-alpine
	docker pull postgres:16-alpine
	docker save redis:7.4-alpine    -o redis.tar
	docker cp redis.tar desktop-control-plane:/tmp/redis.tar
	docker exec desktop-control-plane ctr images import /tmp/redis.tar
	docker exec desktop-control-plane rm /tmp/redis.tar
	rm -f redis.tar
	docker save postgres:16-alpine  -o postgres.tar
	docker cp postgres.tar desktop-control-plane:/tmp/postgres.tar
	docker exec desktop-control-plane ctr images import /tmp/postgres.tar
	docker exec desktop-control-plane rm /tmp/postgres.tar
	rm -f postgres.tar
	@echo "✓ Infra images loaded into k8s node"

build: ## Build all Docker images
	docker build -t devops-agent/collector:latest  ./collector
	docker build -t devops-agent/backend:latest    ./backend
	docker build -t devops-agent/agent:latest      ./agent
	docker build -t devops-agent/dashboard:latest  ./dashboard
	@echo "✓ All images built"

build-minikube: ## Build images in minikube Docker daemon
	eval $$(minikube docker-env) && \
	docker build -t devops-agent/collector:latest  ./collector && \
	docker build -t devops-agent/backend:latest    ./backend && \
	docker build -t devops-agent/agent:latest      ./agent && \
	docker build -t devops-agent/dashboard:latest  ./dashboard
	@echo "✓ All images built (minikube)"

build-ml: ## Build ML model server image (Docker Desktop)
	docker build -t devops-agent/ml-server:latest -f ml-models/serving/Dockerfile ./ml-models
	docker save devops-agent/ml-server:latest -o ml-server.tar
	docker cp ml-server.tar desktop-control-plane:/tmp/ml-server.tar
	docker exec desktop-control-plane ctr images import /tmp/ml-server.tar
	docker exec desktop-control-plane rm /tmp/ml-server.tar
	rm -f ml-server.tar
	@echo "✓ ML server image built and loaded into k8s node"

# ── Deploy ───────────────────────────────────────────────────
.PHONY: deploy deploy-ml deploy-infra deploy-apps switch-claude switch-ml

create-secrets: ## Create or refresh devops-secrets from .env file
	kubectl create secret generic devops-secrets \
	  --namespace $(NAMESPACE) \
	  --from-literal=ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
	  --from-literal=POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
	  --from-literal=POSTGRES_USER=$(POSTGRES_USER) \
	  --from-literal=REDIS_PASSWORD=$(REDIS_PASSWORD) \
	  --dry-run=client -o yaml | kubectl apply -f -

deploy-infra: ## Deploy infrastructure (Redis, Postgres, VictoriaMetrics)
	kubectl apply -f k8s/01-namespace-rbac.yaml
	$(MAKE) create-secrets
	kubectl apply -f k8s/02-redis-pubsub.yaml
	kubectl apply -f k8s/03-postgres.yaml
	kubectl apply -f k8s/03a-victoriametrics.yaml
	kubectl wait --for=condition=ready pod -l app=redis -n $(NAMESPACE) --timeout=120s
	kubectl wait --for=condition=ready pod -l app=postgres -n $(NAMESPACE) --timeout=120s
	kubectl wait --for=condition=ready pod -l app=victoriametrics -n $(NAMESPACE) --timeout=120s
	@echo "✓ Infrastructure deployed"

patch-prometheus: ## Patch external Prometheus to remote_write to VictoriaMetrics (optional)
	kubectl apply -f k8s/prometheus-remote-write-patch.yaml
	@echo "✓ Prometheus remote_write patch applied"

deploy-apps: ## Deploy application services (collector, backend, agent, dashboard)
	kubectl apply -f k8s/04-collector.yaml
	kubectl apply -f k8s/05-backend.yaml
	kubectl apply -f k8s/06-agent.yaml
	kubectl apply -f k8s/07-dashboard.yaml
	@echo "✓ Applications deployed"

deploy: deploy-infra deploy-apps ## Deploy full stack (Claude mode)
	@echo "✓ Full stack deployed (Claude API mode)"

deploy-ml: deploy-infra deploy-apps ## Deploy full stack with ML model server
	kubectl apply -f k8s/08-ml-server.yaml
	kubectl set env deploy/devops-ai-agent -n $(NAMESPACE) ANALYZER_MODE=ml
	@echo "✓ Full stack deployed (ML mode)"

switch-claude: ## Switch agent to Claude API mode
	kubectl set env deploy/devops-ai-agent -n $(NAMESPACE) ANALYZER_MODE=claude
	@echo "✓ Switched to Claude API mode"

switch-ml: ## Switch agent to ML model server mode
	kubectl set env deploy/devops-ai-agent -n $(NAMESPACE) ANALYZER_MODE=ml
	@echo "✓ Switched to ML model server mode"

# ── ML Pipeline ──────────────────────────────────────────────
.PHONY: train train-anomaly train-forecast train-rootcause train-logs test-ml collect-data

train: ## Train all ML models
	cd ml-models && python -m pipeline.train_all

train-anomaly: ## Train anomaly detection model only
	cd ml-models && python -m pipeline.train_all --model anomaly

train-forecast: ## Train forecasting model only
	cd ml-models && python -m pipeline.train_all --model forecasting

train-rootcause: ## Train root cause classifier only
	cd ml-models && python -m pipeline.train_all --model root_cause

train-logs: ## Train log clustering model only
	cd ml-models && python -m pipeline.train_all --model log_clustering

test-ml: ## Run ML pipeline smoke tests
	cd ml-models && python -m pipeline.smoke_test

collect-data: ## Collect real data from running cluster (24h)
	cd ml-models && python -m data.generators.collect_real_data --hours 24

collect-retrain: ## Collect real data and retrain models
	cd ml-models && python -m data.generators.collect_real_data --hours 168 --retrain

# ── Monitoring ───────────────────────────────────────────────
.PHONY: status logs-agent logs-collector logs-backend logs-ml dashboard port-forward

status: ## Show all pods and services
	@echo " Pods"
	@kubectl get pods -n $(NAMESPACE) -o wide
	@echo ""
	@echo "Services"
	@kubectl get svc -n $(NAMESPACE)

logs-agent: ## Stream agent logs
	kubectl logs -f deploy/devops-ai-agent -n $(NAMESPACE)

logs-collector: ## Stream collector logs
	kubectl logs -f deploy/devops-collector -n $(NAMESPACE)

logs-backend: ## Stream backend logs
	kubectl logs -f deploy/devops-backend -n $(NAMESPACE)

logs-ml: ## Stream ML server logs
	kubectl logs -f deploy/devops-ml-server -n $(NAMESPACE)

dashboard: ## Port-forward dashboard (http://localhost:3000)
	@echo "Dashboard: http://localhost:3001"
	kubectl port-forward svc/devops-dashboard -n $(NAMESPACE) 3001:3000

port-forward: ## Port-forward backend API (http://localhost:8000)
	@echo "Backend API: http://localhost:8000"
	kubectl port-forward svc/devops-backend -n $(NAMESPACE) 8000:8000

ml-server: ## Port-forward ML server (http://localhost:8001)
	@echo "ML Server: http://localhost:8001"
	kubectl port-forward svc/devops-ml-server -n $(NAMESPACE) 8001:8001

# ── Cleanup ──────────────────────────────────────────────────
.PHONY: clean clean-ml

clean: ## Tear down all resources
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	@echo "Namespace $(NAMESPACE) deleted"

clean-ml: ## Remove ML server only
	kubectl delete -f k8s/08-ml-server.yaml --ignore-not-found
	kubectl set env deploy/devops-ai-agent -n $(NAMESPACE) ANALYZER_MODE=claude
	@echo "ML server removed, switched to Claude mode"
