import os
import shutil

'''pre-process the dataset such that dsf directories and mit file extensions are discarded.'''
path = os.getcwd()
path = path + "/by_merge/by_merge"
labels = os.listdir(path)
for label in labels:
    subpath = os.path.join(path,label)
    files = os.listdir(subpath)
    to_delete = [file for file in files if not file.endswith(".png")]
    for file in to_delete:
        path_to_file = os.path.join(subpath,file)
        os.remove(path_to_file)
    hsfs = os.listdir(subpath)
    for hsf in hsfs:
        subsubpath = os.path.join(subpath, hsf)
        files = os.listdir(subsubpath)
        for picture in files:
            original = os.path.join(subsubpath, picture)
            destination = os.path.join(subpath, picture)
            shutil.move(original, destination)
        os.rmdir(os.path.join(subpath, hsf))

