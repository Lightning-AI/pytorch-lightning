requirement_fnames = (
    'requirements.txt',
    'requirements/extra.txt',
    'requirements/loggers.txt',
    'requirements/test.txt',
    'requirements/examples.txt',
)


def replace_min(fname: str):
    req = open(fname).read().replace('>=', '==')
    open(fname, 'w').write(req)


if __name__ == '__main__':
    for fname in requirement_fnames:
        replace_min(fname)
