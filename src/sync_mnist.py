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
    length = 200
    image_size = 256
    digit_size = 28
    batch_size = 5
    digit1 = train_dataset[:batch_size]
    digit2 = train_dataset[batch_size : batch_size * 2]
    digit3 = train_dataset[batch_size * 2 : batch_size * 3]
    xs1, ys1, event = get_random_trajectory(
        batch_size, length, image_size, digit_size, move_bound, event=None
    )
    images1 = get_images(batch_size, length, image_size, xs1, ys1, digit1, digit_size)
    xs2, ys2, _ = get_random_trajectory(
        batch_size, length, image_size, digit_size, move_round, event=event
    )
    images2 = get_images(batch_size, length, image_size, xs2, ys2, digit2, digit_size)

    np.random.shuffle(event)
    xs3, ys3, _ = get_random_trajectory(
        batch_size, length, image_size, digit_size, move_round, event=event
    )
    images3 = get_images(batch_size, length, image_size, xs3, ys3, digit3, digit_size)

    images_cat = np.concatenate((images1, images2), axis=3)
    images_flat = np.reshape(images_cat, (-1, image_size, image_size * 2))

    images_pil = []
    for image in images_flat:
        images_pil.append(Image.fromarray(image).convert("P"))
    images_pil[0].save(
        "data/image_sync.gif",
        save_all=True,
        append_images=images_pil[1:],
        optimize=False,
        duration=40,
        loop=0,
    )

    images_cat = np.concatenate((images1, images3), axis=3)
    images_flat = np.reshape(images_cat, (-1, image_size, image_size * 2))

    images_pil = []
    for image in images_flat:
        images_pil.append(Image.fromarray(image).convert("P"))
    images_pil[0].save(
        "data/image_not_sync.gif",
        save_all=True,
        append_images=images_pil[1:],
        optimize=False,
        duration=40,
        loop=0,
    )


def get_random_trajectory(batch_size, length, image_size, digit_size, move_fn, event):
    canvas_size = image_size - digit_size - 1
    trajectory_x = np.zeros((length, batch_size))
    trajectory_y = np.zeros((length, batch_size))
    trajectory_o = np.zeros((length, batch_size), dtype=np.int32)
    if event is None:
        event = np.zeros((length, batch_size), dtype=np.int32)

    for i, (x, y, o) in enumerate(move_fn(length, batch_size, event)):
        trajectory_x[i, :] = x
        trajectory_y[i, :] = y
        trajectory_o[i, :] = o

    trajectory_x = (trajectory_x * canvas_size).astype(np.int32)
    trajectory_y = (trajectory_y * canvas_size).astype(np.int32)

    return trajectory_x, trajectory_y, trajectory_o


def move_bound(length, batch_size, event):
    theta = np.random.rand(batch_size) * 2 * np.pi
    x = np.random.rand(batch_size)
    y = np.random.rand(batch_size)
    step_length = 0.05
    v_x = np.sin(theta)
    v_y = np.cos(theta)

    for i in range(length):
        out_event = np.zeros((batch_size,), dtype=np.int32)
        for j in range(batch_size):
            if event[i, j] == 1:
                v_x[j] = v_x[j] * -1
                v_y[j] = v_y[j] * -1
            x[j] += v_x[j] * step_length
            y[j] += v_y[j] * step_length

            if x[j] <= 0:
                x[j] = 0
                v_x[j] = -v_x[j]
                out_event[j] = 1
            elif x[j] >= 1:
                x[j] = 1
                v_x[j] = -v_x[j]
                out_event[j] = 1

            if y[j] <= 0:
                y[j] = 0
                v_y[j] = -v_y[j]
                out_event[j] = 1
            elif y[j] >= 1:
                y[j] = 1
                v_y[j] = -v_y[j]
                out_event[j] = 1

        yield x, y, out_event


def move_round(length, batch_size, event):
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_theta = np.ones((batch_size,)) * 10 / 180 * np.pi

    for i in range(length):
        x = np.zeros((batch_size,))
        y = np.zeros((batch_size,))
        out_event = np.zeros((batch_size,), dtype=np.int32)
        for j in range(batch_size):
            if event[i, j] == 1:
                v_theta[j] = -v_theta[j]
            theta[j] += v_theta[j]
            x[j] = np.sin(theta[j]) / 2 + 0.5
            y[j] = np.cos(theta[j]) / 2 + 0.5
            if x[j] >= 0.9:
                out_event[j] = 1
        yield x, y, out_event


# print([(x, y) for x, y in move_bound(10, 2)])
main()
