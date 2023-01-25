# Colours
RED    := $(shell tput -Txterm setaf 1)
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

TARGET_MAX_CHAR_NUM=30

## Show help
help:
	@printf "\n\nUsage:\n";
	@printf "${YELLOW}make${RESET} ${GREEN}<target>${RESET}\n\n";

	@printf "Targets:\n";
	@awk '{ \
			if ($$0 ~ /^.PHONY: [a-zA-Z\-\_0-9]+$$/) { \
				helpCommand = substr($$0, index($$0, ":") + 2); \
				if (helpMessage) { \
					printf "\033[33m%-20s \033[32m%s\033[0m\n", \
						helpCommand, helpMessage; \
					helpMessage = ""; \
				} \
			} else if ($$0 ~ /^[a-zA-Z\-\_0-9.]+:/) { \
				helpCommand = substr($$0, 0, index($$0, ":") - 1); \
				if (helpMessage) { \
					printf "\033[33m%-20s \033[32m%s\033[0m\n", \
						helpCommand, helpMessage; \
					helpMessage = ""; \
				} \
			} else if ($$0 ~ /^##/) { \
				if (helpMessage) { \
					helpMessage = helpMessage"\n                     "substr($$0, 3); \
				} else { \
					helpMessage = substr($$0, 3); \
				} \
			} else { \
				if (helpMessage) { \
					printf "\n                     \033[36m%s\033[0m\n\n", helpMessage; \
				} \
				helpMessage = ""; \
			} \
		}' \
		$(MAKEFILE_LIST)
