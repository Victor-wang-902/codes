import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import ada_binarization, noise_remove, skel, vanilla


def preprocessing_1(path):
    original = cv.imread(path)
    img = np.array(original, np.uint8)
    greyscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = ada_binarization(greyscale)
    noise_removed = noise_remove(binary, 20)
    return noise_removed


def histogram_analysis(img, axis=0, mode="plot"):
    binarized = img.copy()
    binarized[binarized == 0] = 1
    binarized[binarized == 255] = 0
    projection = np.sum(binarized, axis=axis)
    height, width = binarized.shape
    blank = np.zeros((height, width, 1), np.uint8)
    if axis:
        for row in range(height):
            cv.line(blank, (0, row), (int(projection[row]), row), 255, 1)
    else:
        for column in range(width):
            cv.line(blank, (column, height), (column, height - int(projection[column])), 255, 1)
    if mode == "plot":
        return blank
    elif mode == "segment":
        return projection


def histogram_analysis_with_skel(img, axis=0, mode="plot"):
    skel_img = skel(img)
    binarized = skel_img
    binarized[binarized == 0] = 1
    binarized[binarized == 255] = 0
    projection = np.sum(binarized, axis=axis)
    height, width = binarized.shape
    blank = np.zeros((height, width, 1), np.uint8)
    if axis:
        for row in range(height):
            cv.line(blank, (0, row), (int(projection[row]), row), 255, 1)
    else:
        for column in range(width):
            cv.line(blank, (column, height), (column, height - int(projection[column])), 255, 1)
    if mode == "plot":
        return blank
    elif mode == "segment":
        return projection


def character_segmentation(img, projection, count_threshold=0, consecutive_threshold=5, mode="plot"):
    height, width = img.shape
    to_average = 0
    consecutive_count = 0
    seg_lines = []
    nmuloc = width - 1
    while np.mean(img[:, nmuloc]) == 255.:
        nmuloc -= 1
    lower_bound = nmuloc + consecutive_threshold
    for column in range(width):
        if projection[column] <= count_threshold:
            to_average += column
            consecutive_count += 1
        else:
            if consecutive_count >= consecutive_threshold:
                average = to_average // consecutive_count
                seg_lines.append(average)
            to_average = 0
            consecutive_count = 0

    seg_lines.append(lower_bound)
    if mode == "plot":
        new_image = img.copy()
        for line in seg_lines:
            cv.line(new_image, (line, 0), (line, height), 0, 2)
        return new_image
    if mode == "segment":
        return seg_lines


def character_segmentation_1(img, projection, count_threshold=1, min_threshold=30, mode="plot"):  # this doen't work
    # without machine learning approach
    height, width = img.shape
    to_average = 0
    consecutive_zeros_count = 0
    seg_lines = []
    nmuloc = width - 1
    PSC = []
    SC = []
    while np.mean(img[:, nmuloc]) == 255.:
        nmuloc -= 1
    lower_bound = nmuloc + count_threshold
    for column in range(width):
        if projection[column] == 0:
            consecutive_zeros_count += 1
        else:
            if consecutive_zeros_count != 0:
                SC.append(list(range(column - consecutive_zeros_count, column + 1)))
                consecutive_zeros_count = 0
            elif projection[column] <= count_threshold:
                PSC.append(column)
    for j, column in enumerate(PSC):
        dist = float("inf")
        to_merge = None
        for i, zero_column in enumerate(SC):
            temp = min(column - zero_column[0], column - zero_column[-1])
            if dist > temp:
                dist = temp
                to_merge = i
        if to_merge is not None:
            if dist <= min_threshold:
                SC[i].append(column)
                PSC[j] = -1
    for column in PSC:
        if column == -1:
            continue
        dist = float("inf")
        to_merge = None
        for i, zero_column in enumerate(SC):
            temp = min(column - zero_column[0], column - zero_column[-1])
            if dist > temp:
                dist = temp
                to_merge = i
        if to_merge is not None:
            if dist <= min_threshold:
                SC[i].append(column)
            else:
                SC.append([column])
        else:
            SC.append([column])
    for columns in SC:
        seg_lines.append(int(np.mean(columns)))
    seg_lines.append(lower_bound)
    if mode == "plot":
        new_image = img.copy()
        for line in seg_lines:
            cv.line(new_image, (line, 0), (line, height), 0, 2)
        return new_image
    if mode == "segment":
        return seg_lines


def word_segmentation(img, projection, count_threshold=0, consecutive_threshold=60, mode="plot"):
    height, width = img.shape
    to_average = 0
    consecutive_count = 0
    seg_lines = []
    nmuloc = width - 1
    lower_bound = nmuloc
    while np.mean(img[:, nmuloc]) == 255.:
        nmuloc -= 1
        lower_bound = nmuloc
    for column in range(width):
        if projection[column] <= count_threshold:
            to_average += column
            consecutive_count += 1
        else:
            if consecutive_count >= consecutive_threshold:
                average = to_average // consecutive_count
                seg_lines.append(average)
            to_average = 0
            consecutive_count = 0

    seg_lines.append(lower_bound)
    if mode == "plot":
        new_image = img.copy()
        for line in seg_lines:
            cv.line(new_image, (line, 0), (line, height), 0, 2)
        return new_image
    if mode == "segment":
        return seg_lines


def line_segmentation(img, projection, count_threshold=5, consecutive_threshold=30, mode="plot"):
    height, width = img.shape
    to_average = 0
    consecutive_count = 0
    seg_lines = []
    wor = height - 1
    lower_bound = wor
    while np.mean(img[wor]) == 255.:
        wor -= 1
        lower_bound = wor
    second_line = False
    for row in range(height):
        if row > count_threshold and not second_line:
            second_line = True
            first_pixel = row
        if second_line:
            if row <= count_threshold:
                consecutive_threshold = row - first_pixel
    for row in range(height):
        if projection[row] <= count_threshold:
            to_average += row
            consecutive_count += 1
        else:
            if consecutive_count >= consecutive_threshold:
                average = to_average // consecutive_count
                seg_lines.append(average)
            to_average = 0
            consecutive_count = 0

    seg_lines.append(lower_bound)
    if mode == "plot":
        print(seg_lines)
        new_image = img.copy()
        for line in seg_lines:
            cv.line(new_image, (0, line), (width, line), 0, 2)
        return new_image
    if mode == "segment":
        return seg_lines


def segment_generator(seg_lines, img, axis):
    num_subimg = len(seg_lines) - 1
    seg_lines.sort()
    if axis == 0:
        for i in range(num_subimg):
            yield img[:, seg_lines[i]:seg_lines[i+1]]
    elif axis == 1:
        for i in range(num_subimg):
            yield img[seg_lines[i]:seg_lines[i+1], :]


def segmentation_pipeline(img_path, method):
    lines = []
    processed = preprocessing_1(img_path)
    line_projection = histogram_analysis_with_skel(processed, axis=1, mode="segment")
    line_seg_lines = line_segmentation(processed, line_projection, mode="segment")
    for sentence in segment_generator(line_seg_lines, processed, axis=1):
        word_projection = histogram_analysis_with_skel(sentence, axis=0, mode="segment")
        word_seg_lines = word_segmentation(sentence, word_projection, mode="segment")
        words = []
        for word in segment_generator(word_seg_lines, sentence, axis=0):
            ch_projection = histogram_analysis_with_skel(word, axis=0, mode="segment")
            ch_seg_lines = character_segmentation(word, ch_projection, mode="segment")
            chs = []
            for ch in segment_generator(ch_seg_lines, word, axis=0):
                ch_processed = method(ch, 177, 28, 20)
                chs.append(ch_processed)
            words.append(chs)
        lines.append(words)
    return lines


def plot_vertical_ch(img_path):
    preprocessed_img = preprocessing_1(img_path)
    hist_original = histogram_analysis(preprocessed_img, axis=0)
    hist_skel = histogram_analysis_with_skel(preprocessed_img, axis=0)
    skel_img = skel(preprocessed_img)
    projection = histogram_analysis_with_skel(skel_img, axis=0, mode="segment")
    seg_img = character_segmentation(skel_img, projection)
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_title("original")
    ax1.imshow(preprocessed_img, "gray", vmin=0, vmax=255)
    ax2 = plt.subplot(3, 2, 2)
    ax2.set_title("histogram for original image")
    ax2.imshow(hist_original, "gray", vmin=0, vmax=255)
    ax3 = plt.subplot(3, 2, 3)
    ax3.set_title("skeletonized")
    ax3.imshow(skel_img, "gray", vmin=0, vmax=255)
    ax4 = plt.subplot(3, 2, 4)
    ax4.set_title("histogram for skeletonized image")
    ax4.imshow(hist_skel, "gray", vmin=0, vmax=255)
    ax5 = plt.subplot(3, 2, 5)
    ax5.imshow(seg_img, "gray", vmin=0, vmax=255)
    plt.show()


def plot_vertical_words(img_path):
    preprocessed_img = preprocessing_1(img_path)
    hist_original = histogram_analysis(preprocessed_img, axis=0)
    hist_skel = histogram_analysis_with_skel(preprocessed_img, axis=0)
    skel_img = skel(preprocessed_img)
    projection = histogram_analysis_with_skel(skel_img, axis=0, mode="segment")
    seg_img = word_segmentation(skel_img, projection)
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_title("original")
    ax1.imshow(preprocessed_img, "gray", vmin=0, vmax=255)
    ax2 = plt.subplot(3, 2, 2)
    ax2.set_title("histogram for original image")
    ax2.imshow(hist_original, "gray", vmin=0, vmax=255)
    ax3 = plt.subplot(3, 2, 3)
    ax3.set_title("skeletonized")
    ax3.imshow(skel_img, "gray", vmin=0, vmax=255)
    ax4 = plt.subplot(3, 2, 4)
    ax4.set_title("histogram for skeletonized image")
    ax4.imshow(hist_skel, "gray", vmin=0, vmax=255)
    ax5 = plt.subplot(3, 2, 5)
    ax5.imshow(seg_img, "gray", vmin=0, vmax=255)
    plt.show()


def plot_horizontal(img_path):
    preprocessed_img = preprocessing_1(img_path)
    hist_original = histogram_analysis(preprocessed_img, axis=1)
    hist_skel = histogram_analysis_with_skel(preprocessed_img, axis=1)
    skel_img = skel(preprocessed_img)
    projection = histogram_analysis_with_skel(skel_img, axis=1, mode="segment")
    seg_img = line_segmentation(skel_img, projection)
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(preprocessed_img, "gray", vmin=0, vmax=255)
    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(hist_original, "gray", vmin=0, vmax=255)
    ax3 = plt.subplot(3, 2, 3)
    ax3.imshow(skel_img, "gray", vmin=0, vmax=255)
    ax4 = plt.subplot(3, 2, 4)
    ax4.imshow(hist_skel, "gray", vmin=0, vmax=255)
    ax5 = plt.subplot(3, 2, 5)
    ax5.imshow(seg_img, "gray", vmin=0, vmax=255)
    plt.show()


if __name__ == "__main__":
    horizontal_img_path = os.path.join(os.getcwd(), "task.png")
    plot_horizontal(horizontal_img_path)
    vertical_words_img_path = os.path.join(os.getcwd(), "task_1.png")
    plot_vertical_words(vertical_words_img_path)
    vertical_ch_img_path = os.path.join(os.getcwd(), "task_2.png")
    plot_vertical_ch(vertical_ch_img_path)
