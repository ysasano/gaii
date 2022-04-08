import datetime
import os
import fire


def stat(filepath):
    mtime = os.path.getatime(filepath)
    print(datetime.datetime.fromtimestamp(mtime))


if __name__ == "__main__":
    fire.Fire(stat)
