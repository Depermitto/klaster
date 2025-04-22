MAKEFLAGS += --silent

doc:
	source .venv/bin/activate; \
	mkdocs build; \
	cd rust; \
	cargo doc --release
