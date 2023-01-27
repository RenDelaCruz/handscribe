# Colours
RED    := $(shell tput -Txterm setaf 1)
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

## Show help
help:
	@printf "\n\n";
	clear
	@printf "Usage:\n";
	@printf "${YELLOW}make${RESET} ${GREEN}<target>${RESET}\n\n";

	@printf "Targets:\n";
	@awk '{ \
			if ($$0 ~ /^.PHONY: [a-zA-Z\-\_0-9]+$$/) { \
				command = substr($$0, index($$0, ":") + 2); \
				if (message) { \
					printf "\033[33m%-20s \033[32m%s\033[0m\n", command, message; \
					message = ""; \
				} \
			} else if ($$0 ~ /^[a-zA-Z\-\_0-9.]+:/) { \
				command = substr($$0, 0, index($$0, ":") - 1); \
				if (message) { \
					printf "\033[33m%-20s \033[32m%s\033[0m\n", command, message; \
					message = ""; \
				} \
			} else if ($$0 ~ /^##/) { \
				if (message) { \
					message = message"\n                     "substr($$0, 4); \
				} else { \
					message = substr($$0, 4); \
				} \
			} else if (message) { \
				lineSkip = command ? "\n" : ""; \
				printf "%s\033[36m[%s]\033[0m\n", lineSkip, message; \
				message = ""; \
			} \
		}' \
		$(MAKEFILE_LIST)
