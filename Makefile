CAP ?= example
IMAGE = ghcr.io/rammp-org/rammp-kitchen-capability-engines/$(CAP)
TAG ?= dev
ROS_DOMAIN_ID ?= 0
SHEPPY = sheppy @ git+https://github.com/rammp-org/sheppy@main
# STANDALONE=1 runs without the orchestrator; MOCK=1 skips hardware.
STANDALONE ?= 0
MOCK ?= 0
LAUNCH_ARGS = use_orchestrator:=$(if $(filter 1,$(STANDALONE)),false,true) \
	mock:=$(if $(filter 1,$(MOCK)),true,false)
ACTION ?=
PARAMS ?= {}

.PHONY: new build run action check smoke lint test-arm

# Start a capability from the example: make new CAP=button_press
new:
	@test ! -e capabilities/$(CAP) || (echo "capabilities/$(CAP) already exists" && exit 1)
	cp -r capabilities/example capabilities/$(CAP)
	cd capabilities/$(CAP) && mv example $(CAP) && mv resource/example resource/$(CAP) \
		&& mv launch/example.launch.py launch/$(CAP).launch.py
	grep -rl -i example capabilities/$(CAP) | xargs sed -i -e 's/example/$(CAP)/g' -e 's/Example/$(CAP)/g'
	printf '/capabilities/%s/  @%s\n' $(CAP) "$$(gh api user --jq .login 2>/dev/null || echo OWNER)" \
		>> .github/CODEOWNERS
	@echo "Created capabilities/$(CAP). Check the owner line in .github/CODEOWNERS."

build:
	docker build -f capabilities/$(CAP)/Dockerfile -t $(IMAGE):$(TAG) .

run:
	docker run --rm -it --network host --ipc host -e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
		$(IMAGE):$(TAG) ros2 launch $(CAP) $(CAP).launch.py $(LAUNCH_ARGS)

# Send one action to a running capability, as the orchestrator would:
#   make action CAP=example ACTION=count PARAMS='{"to": 5}'
action:
	docker run --rm --network host --ipc host -e ROS_DOMAIN_ID=$(ROS_DOMAIN_ID) \
		$(IMAGE):$(TAG) ros2 action send_goal --feedback /$(CAP)/execute_action \
		rammp_orchestrator_interfaces/action/ExecuteAction \
		'{engine_id: $(CAP), action_id: $(ACTION), params_json: "$(subst ",\",$(PARAMS))"}'

check:
	uv run --no-project --with pyyaml --with pytest python3 -m pytest -q tests
	uv run --no-project --with pyyaml --with "$(SHEPPY)" python3 scripts/check_capabilities.py

smoke:
	uv run --no-project --with pyyaml python3 scripts/smoke.py $(CAP) $(IMAGE):$(TAG)

lint:
	pre-commit run --all-files

# self.arm tests, in an x86 ROS 2 Humble image with the repo mounted at /repo (edits need
# no rebuild). Directories that do not exist yet are skipped. ARGS replaces the paths:
#   make test-arm ARGS=tests/arm/test_sim.py -k golden
ARM_TEST_IMAGE = rammp-arm-test:dev
ARM_TEST_DIRS = $(wildcard capability_engine/test tests/arm)
test-arm:
	docker build -f docker/test.Dockerfile -t $(ARM_TEST_IMAGE) docker
	docker run --rm --user $$(id -u):$$(id -g) -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
		-v $(CURDIR):/repo $(ARM_TEST_IMAGE) python3 -m pytest -q -p no:cacheprovider $(or $(ARGS),$(ARM_TEST_DIRS))
