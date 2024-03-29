import datetime
from functools import partial
from pathlib import Path

import fire
import torch
import visualize
import gaii_joint_cnn

from functools import partial
import numpy as np

mini = False


def create_experiment_dir(now, name):
    experiment_dir = Path("data/exp_gaii_cnn") / now / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def load_model():
    model_fn = partial(
        gaii_joint_cnn.fit_q,
        use_time_invariant_term=True,
        length=8,
    )
    return model_fn


def load_images():
    return np.load("data/sync_mnist.npz")


def save_and_visualize_model(model, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model["G"].state_dict(), model_dir / "generator.pth")
    torch.save(model["D"].state_dict(), model_dir / "discriminator.pth")
    visualize.visualize_model(model, model_dir)


def normalize(image):
    """(-1, 1)に正規化"""
    return (image / 255.0 * 2) - 1


def experiment_gaii():
    n_step = 100 if mini else 10000
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    data_dir1 = create_experiment_dir(now, "gaii_cnn1")
    data_dir2 = create_experiment_dir(now, "gaii_cnn2")
    # GAIIの算出
    model_fn = load_model()
    images = load_images()
    result = model_fn(
        normalize(images["images1"]),
        normalize(images["images2"]),
        n_step=n_step,
        debug=True,
        data_dir=data_dir1,
    )
    # 学習結果の可視化・保存
    save_and_visualize_model(
        model=result,
        model_dir=data_dir1,
    )

    result = model_fn(
        normalize(images["images1"]),
        normalize(images["images3"]),
        n_step=n_step,
        debug=True,
        data_dir=data_dir2,
    )

    # 学習結果の可視化・保存
    save_and_visualize_model(
        model=result,
        model_dir=data_dir2,
    )


if __name__ == "__main__":
    fire.Fire(experiment_gaii)
