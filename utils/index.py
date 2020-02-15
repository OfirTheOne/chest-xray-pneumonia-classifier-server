

import os


def rel_to_abs(file, rel):
    current_dir = os.path.dirname(os.path.abspath(file))
    abs_folder_path = os.path.abspath(os.path.join(current_dir, rel))
    return abs_folder_path
