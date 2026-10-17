CAP ?= example
IMAGE = ghcr.io/rammp-org/rammp-kitchen-capability-engines/$(CAP)
TAG ?= dev
ROS_DOMAIN_ID ?= 0
SHEPPY = sheppy @ git+https://github.com/rammp-org/sheppy@main

.PHONY: new build run check smoke lint

# Start a capability from the example: make new CAP=button_press
new:
	@test ! -e capabilities/$(CAP) || (echo "capabilities/$(CAP) already exists" && exit 1)
	cp -r capabilities/example capabilities/$(CAP)
	mv capabilities/$(CAP)/example_node.py capabilities/$(CAP)/$(CAP)_node.py
	grep -rl example capabilities/$(CAP) | xargs sed -i -e 's/example_node/$(CAP)_node/g' \
		-e 's/example/$(CAP)/g' -e 's/Example/$(CAP)/g'
	printf '/capabilities/%s/  @%s\n' $(CAP) "$$(gh api user --jq .login 2>/dev/null || echo OWNER)" \
		>> .github/CODEOWNERS
	@echo "Created capabilities/$(CAP). Check the owner line in .github/CODEOWNERS."

build:
	docker build -f capabilities/$(CAP)/Dockerfile -t $(IMAGE):$(TAG) .

run:
	docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) $(IMAGE):$(TAG)

check:
	uv run --no-project --with pyyaml --with pytest python3 -m pytest -q tests
	uv run --no-project --with pyyaml --with "$(SHEPPY)" python3 scripts/check_capabilities.py

smoke:
	uv run --no-project --with pyyaml python3 scripts/smoke.py $(CAP) $(IMAGE):$(TAG)

lint:
	pre-commit run --all-files
