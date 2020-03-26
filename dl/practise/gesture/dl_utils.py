import numpy as np
import os
import glob
import math
from dl.practise.gesture.config import *
from PIL import Image
import tensorflow as tf


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


def image_eraser(image):
    """图像随机擦除"""
    eraser_ = get_random_eraser(s_h=0.3, pixel_level=True)
    return eraser_(image)


def delfile(path):
    """删除目录"""
    fileNames = glob.glob(path + r'/*')
    for fileName in fileNames:
        try:
            os.remove(fileName)
        except:
            try:
                os.rmdir(fileName)
            except:
                delfile(fileName)
                os.rmdir(fileName)


def restrict_data(small_list):
    """加载图片训练数据时，限制数量，按比例加载"""
    len1 = len(small_list)
    retain_count = math.floor(len1 * DATA_RATE)
    print(f'限制比例： {DATA_RATE}', '保留数据量： ', retain_count, ' 总共数据量： ', len1)
    return small_list[:retain_count]


def read_image_paths(path):
    """文件夹中读取所有图片地址"""
    return [path + filename for filename in os.listdir(path)]


def read_and_restrict_imagepaths(path):
    """文件夹中读取所有图片地址，限制数量（可以指定）"""
    list1 = read_image_paths(path)
    return restrict_data(list1)


def normalization(image_ndarray):
    """归一化处理样本特征值"""
    assert image_ndarray.ndim in (3, 4)
    assert image_ndarray.shape[-1] == 3

    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    image_ndarray = image_ndarray - np.array(MEAN_RGB)
    image_ndarray = image_ndarray / np.array(STDDEV_RGB)

    return image_ndarray


def mixup(batch_x, batch_y, alpha=0.2):
    # 警告：这个方法有问题，暂时别使用
    """数据混合mixup"""
    # size = self.batch_size
    size = batch_y.shape[0]
    l = np.random.beta(alpha, alpha, size)

    X_l = l.reshape(size, 1, 1, 1)
    y_l = l.reshape(size)
    # y_l = l.reshape(size, 1)

    X1 = batch_x
    Y1 = batch_y
    X2 = batch_x[::-1]
    Y2 = batch_y[::-1]

    X = X1 * X_l + X2 * (1 - X_l)
    Y = Y1 * y_l + Y2 * (1 - y_l)

    return X, Y


def get_batch_data(narray, idx, batch_size):
    """获取一批次数据"""
    return narray[idx * batch_size:(idx + 1) * batch_size]


def read_image_and_resize(img_path, img_with_or_height):
    """读取出图片，并且按比例缩放"""
    img = Image.open(img_path)
    # [180, 200, 3]
    # scale = self.img_size[0] / max(img.size[:2])
    scale = img_with_or_height / max(img.size[:2])
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
    img = img.convert('RGB')
    img = np.array(img)
    return img


def x_y_hstack(img_paths, labels):
    """测试数据与标签值合并为一个ndarray数据"""
    return np.hstack((np.array(img_paths).reshape(len(img_paths), 1),
                      np.array(labels).reshape(len(labels), 1)))


def image_enhance(image):
    """图片数据增加：反转，移动"""
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        vertical_flip=True,
    )
    image = datagen.random_transform(image)
    return image


def image_fill_padding(img, size=None, fill_value=255):
    """改变图片尺寸到NxN，并且做填充使得图像处于中间位置"""
    h, w = img.shape[:2]
    if size is None:
        size = max(h, w)
    shape = (size, size) + img.shape[2:]
    background = np.full(shape, fill_value, np.uint8)
    center_x = (size - w) // 2
    center_y = (size - h) // 2
    background[center_y:center_y + h, center_x:center_x + w] = img
    return background
