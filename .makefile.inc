# Colours
RED    := $(shell tput -Txterm setaf 1)
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
BLUE   := $(shell tput -Txterm setaf 6)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

define HELP_SCRIPT
{
	if ($$0 ~ /^.PHONY: [a-zA-Z\-\_0-9]+$$/) {
		command = substr($$0, index($$0, ":") + 2);
		if (message) {
			printf "${YELLOW}%-20s ${GREEN}%s${RESET}\n", command, message;
			message = "";
		}
	} else if ($$0 ~ /^[a-zA-Z\-\_0-9.]+:/) {
		command = substr($$0, 0, index($$0, ":") - 1);
		if (message) {
			printf "${YELLOW}%-20s ${GREEN}%s${RESET}\n", command, message;
			message = "";
		}
	} else if ($$0 ~ /^##/) {
		if (message) {
			message = sprintf("%s\n                     \t%s", message, substr($$0, 4));
		} else {
			message = substr($$0, 4);
		}
	} else if (message) {
		lineSkip = command ? "\n" : "";
		printf "%s${BLUE}[%s]${RESET}\n", lineSkip, message;
		message = "";
	}
}
endef
export HELP_SCRIPT

## Show help
help:
	@printf "\n\n";
	@clear
	@printf "Usage:\n";
	@printf "${YELLOW}make${RESET} ${GREEN}<target>${RESET}\n\n";

	@printf "Targets:\n";
	@awk "$${HELP_SCRIPT}" $(MAKEFILE_LIST)
