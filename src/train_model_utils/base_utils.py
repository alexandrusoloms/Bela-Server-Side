import os


def make_path(path_name):
    """
    creates a path to a directory on the machine.
    :param path_name: (str) path from current directory
    :return: str -> full path
    """
    return os.path.join(os.path.dirname("__file__"), "", ) + path_name