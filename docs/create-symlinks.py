import glob
import os


def symlink_folder(source_dir, target_dir: str = "source-lit") -> None:
    assert os.path.isdir(source_dir)
    assert os.path.isdir(target_dir)
    ls = glob.glob(os.path.join(source_dir, "**"), recursive=True)
    for path_ in ls:
        path_target = path_.replace(source_dir, target_dir)
        if os.path.isdir(path_) or os.path.exists(path_target):
            continue
        if os.path.islink(path_target):
            print(path_target)
            continue
        path_dir = os.path.dirname(path_target)
        os.makedirs(path_dir, exist_ok=True)
        depth = path_.count(os.path.sep)
        path_root = os.path.sep.join([".."] * depth)
        path_source = os.path.join(path_root, path_)
        # print(path_source, path_target, os.path.exists(path_target))
        os.symlink(path_source, path_target)


if __name__ == "__main__":
    for name in ("app", "pytorch"):
        symlink_folder(f"source-{name}")
