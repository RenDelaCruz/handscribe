include .makefile.inc

## Starts the project. Use ESC in the app or CTRL+C in the terminal to stop
.PHONY: start
start:
	poetry run python3 -m src.main

## Initializes the project by installing packages 
.PHONY: init
init:
	poetry install

## Development

## Runs the training script to build the classifier model
.PHONY: learn
learn:
	poetry run python3 -m src.key_train

## Formats the code using black
.PHONY: format
format:
	poetry run black .

## Type checks the project with mypy
.PHONY: type_check
type_check:
	poetry run mypy .

## Auto-lints the code using ruff
.PHONY: lint
lint:
	poetry run ruff . --fix

## Runs all the above formatters
.PHONY: fix
fix:
	$(MAKE) format
	$(MAKE) type_check
	$(MAKE) lint

## Runs unit tests. Usage: make test [path.to.TestCase.method]
.PHONY: test
test:
	poetry run python3 -m unittest $(word 2, $(MAKECMDGOALS))

## Packages

## Adds a package or dev dependency to the project. Usage: make install [dev] <package>
.PHONY: install
install:
ifeq ($(filter dev, $(MAKECMDGOALS)),)
	poetry add $(word 2, $(MAKECMDGOALS))
else
	poetry add --group dev $(word 3, $(MAKECMDGOALS))
endif

## Deletes a package from the project. Usage: make uninstall <package>
.PHONY: uninstall
uninstall:
	poetry remove $(word 2, $(MAKECMDGOALS))

## Other

## Activates virtual environment
.PHONY: activate
activate:
	# source .venv/bin/activate 
	. .venv/bin/activate 

## Deactivates virtual environment
.PHONY: deactivate
deactivate:
	# source deactivate
	. deactivate
