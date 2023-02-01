![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  
[![python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy](https://img.shields.io/badge/type_checker-mypy-teal.svg)](http://mypy-lang.org/)
[![ruff](https://img.shields.io/badge/linter-ruff-red.svg)](http://mypy-lang.org/)


# Sign Language Alphabet Translator  <!-- omit from toc -->

![Demo](docs/assets/demo.gif)

## Table of Contents <!-- omit from toc -->

- [Introduction](#introduction)
  - [American Sign Language](#american-sign-language)
  - [Project Objective](#project-objective)
- [Usage](#usage)
  - [Environment Setup](#environment-setup)
  - [Running the Program](#running-the-program)
  - [Modes](#modes)
  - [Quitting](#quitting)

## Introduction

### American Sign Language

American Sign Language (ASL) serves as the main sign language of the deaf community in North America. In the language, words and grammar are composed of a combination of hand signals and facial expressions to convey sentences. However, for words without a sign, such as names and loanwords, there exists ASL fingerspelling, where English letters can be individually represented with a hand sign.

![Fingerspelling Chart](docs/assets/fingerspelling-chart.png)

For learners of ASL, it may be difficult to find someone from the deaf community to interact with. As such, it is often a challenge receiving adequate practice and feedback, even for something as simple as fingerspelling.

### Project Objective

The Sign Language Alphabet Translator (SLAT) uses machine learning classification algorithms to translate and identify a learner's fingerspelling signs live as it is shown to the computer camera.

## Usage

### Environment Setup

This project requires:
- Python 3.10
- Package manager `poetry`

To install `poetry`, run:

```sh
$ pip install poetry
```

### Running the Program

1. Run `make init` to initialize the project by downloading all dependencies.
2. Run `make start` to start the app.

> Run `make` to see a list of other commands.

### Modes

- Press `1` to toggle the hand landmarks shown on the screen.
- Press `2` to toggle the bounding box.

### Quitting

- Press `ESC` to quit the program.

----

**Capstone Project**  
CS 4ZP6 · Group 24  
Ren de la Cruz (400051394)
