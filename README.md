# Sign Language Alphabet Translator  <!-- omit from toc -->
 
- [Introduction](#introduction)
- [Start](#start)
  - [Environment Setup](#environment-setup)
  - [Running the Project](#running-the-project)

## Introduction

**CS 4ZP6: Capstone Project**  
Ren de la Cruz

## Start

### Environment Setup

This project requires:
- Python 3.10
- Package manager `poetry`

To install `poetry`, run:

```sh
$ pip install poetry
```

### Running the Project

1. Run `make init` to initialize the project.
2. Run `make start` to start the project.

#### Commands

> Run `make` to see a list of all commands.

```makefile
Usage:
make <target>

Targets:
start                Starts the project. Use ESC in the app or CTRL+C in the terminal to stop
init                 Initializes the project by installing packages 

[Development]
format               Formats the code using black
type_check           Type checks the project with mypy
lint                 Auto-lints the code using ruff
fix                  Runs all the above formatters

[Packages]
install              Adds a package or dev dependency to the project. Usage: make install [dev] <package>
uninstall            Deletes a package from the project. Usage: make uninstall <package>

[Other]
activate             Activates virtual environment
deactivate           Deactivates virtual environment
help                 Show help
```
