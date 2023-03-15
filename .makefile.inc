# Colours
RED    := $(shell tput -Txterm setaf 1)
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
BLUE   := $(shell tput -Txterm setaf 6)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

TARGET_MAX_CHAR_NUM = 15

define HELP_SCRIPT
	if ((/^.PHONY: (.*)/ || /^([A-Za-z0-9_-]+):/) && !$$command) {
		$$command = $$1;
	} elsif (/^## (.*)/) {
		if ($$message) {
			$$message = sprintf "%s\n%s%s", $$message, ' ' x ${TARGET_MAX_CHAR_NUM}, $$1;
		} else {
			$$message = $$1;
		}
	} elsif ($$command && $$message) {
		printf "${YELLOW}%-${TARGET_MAX_CHAR_NUM}s${GREEN}%s${RESET}\n", $$command, $$message;
		$$command = ""; $$message = "";
	} elsif ($$message) {
		printf "\n${BLUE}[%s]${RESET}\n", $$message;
		$$message = "";
	}
endef
export HELP_SCRIPT

## Show help
help:
	@clear
	@printf "Usage:\n";
	@printf "${YELLOW}make${RESET} ${GREEN}<target>${RESET}\n\n";

	@printf "Targets:\n";
	@perl -ne "$${HELP_SCRIPT}" $(MAKEFILE_LIST)
