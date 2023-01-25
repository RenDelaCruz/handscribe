include .makefile.inc

## Runs the project through a virtual environment
start:
	poetry run python3 main.py

# ## Activates virtual environment
# activate:
# 	# source .venv/bin/activate 
# 	. .venv/bin/activate 

# ## Deactivates virtual environment
# deactivate:
# 	# source deactivate
# 	. deactivate

## Other

## Initializes the project by installing packages 
install:
	poetry install
