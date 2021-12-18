import os
import cv2 as cv

from preprocess import vanilla, preprocess_procedure_1, preprocess_procedure_2, preprocess_procedure_3, preprocess_procedure_4


'''create a copy of the dataset with processed images'''

def create_processed(method, checkpoint=None):
    working_directory = os.getcwd()
    source_path = working_directory + '/by_merge/by_merge'
    labels = os.listdir(source_path)
    if method == preprocess_procedure_1:
        dir_name = 'preprocessed_style_1'
    elif method == preprocess_procedure_2:
        dir_name = 'preprocessed_style_2'
    elif method == preprocess_procedure_3:
        dir_name = 'preprocessed_style_3'
    elif method == preprocess_procedure_4:
        dir_name = 'preprocessed_style_4'
    elif method == vanilla:
        dir_name = 'vanillaed'
    else:
        raise Exception("method unidentified")
    if checkpoint:
        dest_path = os.path.dirname(checkpoint)
        current_label = os.path.basename(os.path.normpath(checkpoint))
        resuming_index = labels.index(current_label)
        labels = labels[resuming_index:]
    else:
        os.mkdir(os.path.join(working_directory, dir_name))
        dest_path = os.path.join(working_directory, dir_name)

    for label in labels:
        destination = os.path.join(dest_path, label)
        os.mkdir(destination)
        subpath = os.path.join(source_path,label)
        files = os.listdir(subpath)
        for file in files:
            if file.endswith(".png"):
                path_to_file = os.path.join(subpath,file)
                destination_file = os.path.join(destination,file)
                img = cv.imread(path_to_file)
                img = method(img, 177, 28, 20)
                success = cv.imwrite(destination_file, img)
                if not success:
                    raise Exception("something went wrong")

if __name__ == "__main__":
    checkpoint = os.path.join(os.getcwd(), "preprocessed_style_1")
    checkpoint = os.path.join(checkpoint, "6e")
    create_processed(preprocess_procedure_4)

