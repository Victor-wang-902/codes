import os
import cv2 as cv
import pickle

def saved():
    path = os.path.join(os.getcwd(), "vanillaed")
    labels = os.listdir(path)
    count = 0
    for label in labels:
        subpath = os.path.join(path, label)
        files = os.listdir(subpath)
        for file in files:
            if count % 1000 == 0:
                print(count)
            filepath = os.path.join(subpath,file)
            img = cv.imread(filepath)
            img = img[:-1, :-1]
            cv.imwrite(filepath, img)
            count += 1


def fix_pickle():
    with open("learning_rate_tuning.pickle", "rb") as handle:
        f = pickle.load(handle)
    f[-2] = "tuning learning rate on 10 epochs"
    f.append([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
    with open("learning_rate_tuning.pickle", "wb") as handle:
        pickle.dump(f, handle)
if __name__ == "__main__":
    fix_pickle()

