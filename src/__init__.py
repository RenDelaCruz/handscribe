from termcolor import colored


def _(message: str, /) -> str:
    return colored(message, color="green", on_color="on_black", attrs=["bold"])
