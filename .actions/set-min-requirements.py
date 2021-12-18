requirement_fnames = (
    "requirements.txt",
    "requirements/extra.txt",
    "requirements/loggers.txt",
    "requirements/examples.txt",
    # "requirements/test.txt",  # Don't use old testing packages
)


def replace_min(fname: str):
    req = open(fname).read().replace(">=", "==")
    open(fname, "w").write(req)


if __name__ == "__main__":
    for fname in requirement_fnames:
        replace_min(fname)
