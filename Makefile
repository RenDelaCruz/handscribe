include .makefile.inc

## Starts the project. Use CTRL+C to stop
start:
	poetry run python3 main.py

## Initializes the project by installing packages 
init:
	poetry install

## Development

## Formats the code using black
format:
	poetry run black .

## Type checks the project with mypy
type_check:
	poetry run mypy .

## Auto-lints the code using ruff
lint:
	poetry run ruff . --fix

## Packages

## Adds a package to the project. Usage: make install [dev] <package>
install:
ifeq ($(filter dev, $(MAKECMDGOALS)),)
	poetry add $(word 2, $(MAKECMDGOALS))
else
	poetry add --group dev $(word 3, $(MAKECMDGOALS))
endif

## Deletes a package from the project. Usage: make uninstall <package>
uninstall:
	poetry remove $(word 2, $(MAKECMDGOALS))

## Other

## Activates virtual environment
activate:
	# source .venv/bin/activate 
	. .venv/bin/activate 

## Deactivates virtual environment
deactivate:
	# source deactivate
	. deactivate
