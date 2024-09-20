.PHONY: setup
setup: ## Setup development environment
	gcloud secrets versions access latest --secret modal-toml > ~/.modal.toml
