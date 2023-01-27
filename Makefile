include .makefile.inc

## Starts the project. Use CTRL+C to stop
start:
	poetry run python3 main.py

## Initializes the project by installing packages 
install:
	poetry install

## Other Commands

## Activates virtual environment
activate:
	# source .venv/bin/activate 
	. .venv/bin/activate 

## Deactivates virtual environment
deactivate:
	# source deactivate
	. deactivate
