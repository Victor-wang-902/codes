import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def binarization(img, threshold):  # reduce channel to 1 and binarize each grey scaled pixel.
    ret, thresh = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
    return thresh

def ada_binarization(img):
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, 30)

def noise_remove(img, h):  # h is the parameter that determines how strong the filter is.
    dst = cv.fastNlMeansDenoising(img,None,h,21,7)
    return dst

def thinning(img):
    d_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    e_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    dilation = cv.dilate(img, d_kernel, iterations=1)
    erosion = cv.erode(dilation, e_kernel, iterations=1)
    return erosion

def skel(img):
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    img = cv.bitwise_not(img)
    while True:
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        temp = cv.subtract(img, open)
        eroded = cv.erode(img, element)
        skeleton = cv.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv.countNonZero(img) == 0:
            break
    skeleton = cv.bitwise_not(skeleton)
    return skeleton


def center_and_resize(img, new_size, character_size):

    def center_of_mass(img):  # find the center of mass(black dots) of the image
        h, w = img.shape[0], img.shape[1]
        x = -1
        y = -1
        x_weights = 0
        y_weights = 0
        for i in range(h):
            for j in range(w):
                pixel = img[i,j]
                if pixel == 255:
                    continue
                weight = (255 - pixel) / 255
                if x == -1 and y == -1:
                    x = j
                    x_weights += weight
                    y = i
                    y_weights += weight
                else:
                    x = (x * x_weights + j * weight) / (x_weights + weight)
                    y = (y * y_weights + i * weight) / (y_weights + weight)
                    x_weights += weight
                    y_weights += weight

        if x == -1 and y == -1:
            center_x = int(w * 0.5)
            center_y = int(h * 0.5)
        else:
            center_x = int(x)
            center_y = int(y)

        return center_x, center_y

    def bounding_box(img):  # find the boundaries of the black area
        h, w = img.shape[0], img.shape[1]
        x_min = w-1
        x_max = 0
        y_min = h-1
        y_max = 0

        for i in range(h):
            for j in range(w):
                temp = img[i,j]
                if temp == 255:
                    continue

                x_min = min(j,x_min)
                x_max = max(j,x_max)
                y_min = min(i,y_min)
                y_max = max(i,y_max)

        return x_min,x_max,y_min,y_max

    def cropping_box(img,rect,pt):  # find the smallest square centered at the center of mass
        h, w = img.shape[0], img.shape[1]
        new_h, new_w = rect[3] - rect[2] + 1, rect[1] - rect[0] + 1
        size = max(new_h, new_w)
        min_r = size // 2
        for r in range(size):
            x1 = pt[0] - r
            x2 = pt[0] + r
            y1 = pt[1] - r
            y2 = pt[1] + r

            xc1 = rect[0]
            xc2 = rect[1]
            yc1 = rect[2]
            yc2 = rect[3]

            if x1 > xc1: continue
            if y1 > yc1: continue
            if x2 < xc2: continue
            if y2 < yc2: continue

            min_r = r
            break


        new_x_min = pt[0] - min_r
        new_y_min = pt[1] - min_r
        new_x_max = pt[0] + min_r
        new_y_max = pt[1] + min_r

        if new_x_min < 0 or new_y_min < 0 or new_x_max >= w or new_y_max >= h:
            left = -new_x_min if  new_x_min < 0 else 0
            if left: new_x_min = 0
            new_x_max += left
            right = new_x_max - w + 1 if new_x_max >= w else 0
            bottom = -new_y_min if new_y_min < 0 else 0
            if bottom: new_y_min = 0
            new_y_max += bottom
            top = new_y_max - h + 1 if new_y_max >= h else 0
            img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 255)

        return img[new_y_min:new_y_max,new_x_min:new_x_max]

    def downscale(img,character_size):
        try:
            return cv.resize(img,(character_size,character_size),interpolation=cv.INTER_LINEAR)
        except Exception as e:
            print("duh")
            print(e)
            return np.zeros([20,20,1],dtype=np.uint8)

    def pad(img,shape):
        h,w = img.shape[0],img.shape[1]
        left = (shape-w)//2
        top = (shape-h)//2
        right = left if (shape - w) % 2 == 0 else left + 1
        bottom = top if (shape - h) % 2 == 0 else top + 1
        return cv.copyMakeBorder(img,top,bottom,left,right,cv.BORDER_CONSTANT,None,255)

    pt = center_of_mass(img)
    boundaries = bounding_box(img)
    cropped = cropping_box(img,boundaries,pt)
    scaled = downscale(cropped,character_size)
    padded = pad(scaled,new_size)
    return padded


def preprocess_procedure_1(img,threshold,size,ch_size):  # binarization -> noise remove -> thinning -> cropping
    img = np.array(img) if not isinstance(img,type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = binarization(img, threshold)
    noised = noise_remove(thresh,10)
    thinned = thinning(noised)
    processed = center_and_resize(thinned, size, ch_size)
    return processed


def preprocess_procedure_2(img,threshold,size,ch_size):  # binarization -> noise remove -> skeletonization -> cropping
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh = binarization(img, threshold)
    noised = noise_remove(thresh, 10)
    thinned = skel(noised)
    processed = center_and_resize(thinned, size, ch_size)
    return processed


def preprocess_procedure_3(img, threshold, size, ch_size):  # cropping -> binarization -> noise remove -> thinning
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rescaled = center_and_resize(img, size, ch_size)
    thresh = binarization(rescaled, threshold)
    noised = noise_remove(thresh, 10)
    processed = thinning(noised)
    return processed


def preprocess_procedure_4(img, threshold, size, ch_size):  # cropping -> binarization -> noise remove -> skeletonization
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rescaled = center_and_resize(img, size, ch_size)
    thresh = binarization(rescaled,threshold)
    noised = noise_remove(thresh, 10)
    processed = skel(noised)
    return processed


def vanilla(img, threshold, size, ch_size):
    img = np.array(img) if not isinstance(img, type(np.array([]))) else img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) != 2 else img
    img = binarization(img, threshold)
    img = center_and_resize(img, size, ch_size)
    return img


if __name__ == "__main__":
    img = cv.imread("sample_6.png")
    print(type(img))
    proc1 = preprocess_procedure_1(img, 177, 28, 20)
    proc2 = preprocess_procedure_2(img, 177, 28, 20)
    proc3 = preprocess_procedure_3(img, 177, 28, 20)
    proc4 = preprocess_procedure_4(img, 177, 28, 20)
    original = vanilla(img, 177, 28, 22)
    print(original.shape)
    titles = ['Original Image', 'proc1', 'proc2', 'proc3', 'proc4', 'vanilla']
    images = [img, proc1, proc2, proc3, proc4, original]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
