
def make_tuple(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x
    else:
        return (x,)