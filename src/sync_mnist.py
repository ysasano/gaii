from torchvision import datasets
import numpy as np
from PIL import Image

debug = False

# MNISTデータの取得
# 学習用
train_dataset = datasets.MNIST(
    "./data",  # データの保存先
    train=True,  # 学習用データを取得する
    download=True,  # データが無い時にダウンロードする
).data.numpy()


def get_images(batch_size, length, image_size, xs, ys, digit1, digit2, digit_size, es):
    image = np.ones((batch_size, length, image_size, image_size), dtype=np.uint8) * 255
    for batch_idx in range(batch_size):
        for length_idx in range(length):
            x = xs[length_idx, batch_idx]
            y = ys[length_idx, batch_idx]
            if debug:
                print(
                    x,
                    y,
                    x + digit_size,
                    y + digit_size,
                    image.shape,
                    image[
                        batch_idx, length_idx, x : x + digit_size, y : y + digit_size
                    ].shape,
                    es[length_idx, batch_idx],
                )
            # 白地に黒なので255から引く
            image[batch_idx, length_idx, x : x + digit_size, y : y + digit_size] = (
                255 - digit1[batch_idx]
                if np.isclose(es[length_idx, batch_idx], 1.0)
                else 255 - digit2[batch_idx]
            )
    return image


def generate_es(length, batch_size):
    es = np.zeros((length, batch_size))
    remaining = 10
    value = 0
    for i in range(length):
        print(remaining)
        if remaining < 0:
            remaining = np.random.exponential(20) + 1
            value = 1 - value
        es[i, :] = value
        remaining -= 1
    return es


def resize(images, size):
    images_list = []
    for image_seq in images:
        image_seq_list = []
        for image in image_seq:
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize(size)
            image_seq_list.append(np.array(image_pil))
        images_list.append(np.stack(image_seq_list, axis=0))
    return np.stack(images_list, axis=0)


def main():
    length = 200
    image_size = 64
    digit_size = 28
    batch_size = 5
    digit1 = train_dataset[:batch_size]
    digit2 = train_dataset[batch_size : batch_size * 2]
    xs1, ys1, es2 = get_random_trajectory(
        batch_size, length, image_size, digit_size, move_bound
    )
    es1 = np.zeros((length, batch_size))
    images1 = get_images(
        batch_size, length, image_size, xs1, ys1, digit1, digit1, digit_size, es1
    )
    images1 = resize(images1, (32, 32))

    xs2 = (
        np.ones((length, batch_size), dtype=np.int32)
        * (image_size - digit_size - 1)
        // 2
    )
    ys2 = (
        np.ones((length, batch_size), dtype=np.int32)
        * (image_size - digit_size - 1)
        // 2
    )
    images2 = get_images(
        batch_size, length, image_size, xs2, ys2, digit1, digit2, digit_size, es2
    )
    images2 = resize(images2, (32, 32))
    es3 = generate_es(length, batch_size)
    images3 = get_images(
        batch_size, length, image_size, xs2, ys2, digit1, digit2, digit_size, es3
    )
    images3 = resize(images3, (32, 32))

    images_cat = np.concatenate((images1, images2), axis=3)
    images_flat = np.reshape(images_cat, (-1, 32, 32 * 2))

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
    images_flat = np.reshape(images_cat, (-1, 32, 32 * 2))

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
    np.savez_compressed(
        "data/sync_mnist.npz", images1=images1, images2=images2, images3=images3
    )


def get_random_trajectory(batch_size, length, image_size, digit_size, move_fn):
    canvas_size = image_size - digit_size - 1
    trajectory_x = np.zeros((length, batch_size))
    trajectory_y = np.zeros((length, batch_size))
    trajectory_e = np.zeros((length, batch_size))

    for i, (x, y, e) in enumerate(move_fn(length, batch_size)):
        trajectory_x[i, :] = x
        trajectory_y[i, :] = y
        trajectory_e[i, :] = e

    trajectory_x = (trajectory_x * canvas_size).astype(np.int32)
    trajectory_y = (trajectory_y * canvas_size).astype(np.int32)

    return trajectory_x, trajectory_y, trajectory_e


def move_bound(length, batch_size):
    step = 0.05
    x = np.ones((batch_size,)) * 0.5
    y = np.ones((batch_size,)) * 0.5
    e = np.zeros((batch_size,))
    s = np.ones((batch_size,)) * step

    for _ in range(length):
        for j in range(batch_size):
            y[j] += s[j]
            if y[j] <= 0:
                y[j] = 0
                s[j] = -s[j]
                e[j] = 1 - e[j]
            elif y[j] >= 1:
                y[j] = 1
                s[j] = -s[j]
                e[j] = 1 - e[j]
        yield x, y, e


def move_round(length, batch_size, event):
    x = 0.5
    y = 0.5
    step = 0.05

    for _ in range(length):
        out_event = np.zeros((batch_size,), dtype=np.int32)
        for j in range(batch_size):
            x[j] += step
            if x[j] <= 0:
                x[j] = 0
                step = -step
                out_event[j] = 1
            elif x[j] >= 1:
                x[j] = 1
                step = -step
                out_event[j] = 1

        yield x, y, out_event


# print([(x, y) for x, y in move_bound(10, 2)])
main()
