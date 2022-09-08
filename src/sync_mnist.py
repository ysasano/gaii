from random import sample
from torchvision import datasets
import numpy as np
from PIL import Image


# MNISTデータの取得
# 学習用
train_dataset = datasets.MNIST(
    "./data",  # データの保存先
    train=True,  # 学習用データを取得する
    download=True,  # データが無い時にダウンロードする
).data.numpy()


def sample_sign(length, rparam):
    sign = np.zeros((length, 2))
    past = np.random.choice([1, -1], size=(2,))
    sign[0, :] = past
    for i in range(1, length):
        curr = [0, 0]
        if past[0] == 1 and past[1] == 1:
            curr[0] = 1
            curr[1] = np.random.choice([-1, 1], p=[rparam, 1 - rparam])
        else:
            curr[0] = -1
            curr[1] = np.random.choice([-1, 1], p=[1 - rparam, rparam])
        sign[i, :] = curr
        past = curr
    return sign


def get_images(batch_size, length, image_size, xs, ys, digit, digit_size):
    image = np.ones((batch_size, length, image_size, image_size), dtype=np.uint8) * 255
    for batch_idx in range(batch_size):
        for length_idx in range(length):
            x = xs[length_idx, batch_idx]
            y = ys[length_idx, batch_idx]
            image[batch_idx, length_idx, x : x + digit_size, y : y + digit_size] = (
                255 - digit[batch_idx]
            )
    return image


def main():
    length = 40
    image_size = 64
    digit_size = 28
    batch_size = 5
    rparam = 0.1
    digit1 = train_dataset[:batch_size]
    digit2 = train_dataset[batch_size : batch_size * 2]
    sign = sample_sign(length, rparam)
    xs1, ys1 = get_random_trajectory(
        batch_size, length, image_size, digit_size, sign[:, 0], move_bound
    )
    images1 = get_images(batch_size, length, image_size, xs1, ys1, digit1, digit_size)
    xs2, ys2 = get_random_trajectory(
        batch_size, length, image_size, digit_size, sign[:, 1], move_bound
    )
    images2 = get_images(batch_size, length, image_size, xs2, ys2, digit2, digit_size)
    images_cat = np.concatenate((images1, images2), axis=3)
    images_flat = np.reshape(images_cat, (-1, image_size, image_size * 2))

    images_pil = []
    for image in images_flat:
        images_pil.append(Image.fromarray(image).convert("P"))
    images_pil[0].save(
        "data/image.gif",
        save_all=True,
        append_images=images_pil[1:],
        optimize=False,
        duration=40,
        loop=0,
    )


def get_random_trajectory(batch_size, length, image_size, digit_size, sign, move_fn):
    canvas_size = image_size - digit_size - 1
    trajectory_x = np.zeros((length, batch_size))
    trajectory_y = np.zeros((length, batch_size))

    for i, (x, y) in enumerate(move_fn(length, batch_size, sign)):
        trajectory_x[i, :] = x
        trajectory_y[i, :] = y

    trajectory_x = (trajectory_x * canvas_size).astype(np.int32)
    trajectory_y = (trajectory_y * canvas_size).astype(np.int32)

    return trajectory_x, trajectory_y


def move_bound(length, batch_size, sign):
    theta = np.random.rand(batch_size) * 2 * np.pi
    x = np.random.rand(batch_size)
    y = np.random.rand(batch_size)
    step_length = 0.2
    v_x = np.sin(theta)
    v_y = np.cos(theta)

    for i in range(length):
        x += v_x * step_length * sign[i]
        y += v_y * step_length * sign[i]
        for j in range(batch_size):
            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
            elif x[j] >= 1:
                x[j] = 1
                v_x[j] = -v_x[j]

            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
            elif y[j] >= 1:
                y[j] = 1
                v_y[j] = -v_y[j]
        yield x, y


# print([(x, y) for x, y in move_bound(10, 2)])
# main()
print(sample_sign(100, 0.25))
