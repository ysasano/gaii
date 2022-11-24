from functools import partial
from pathlib import Path

import fire
import torch
import visualize
import gaii_joint_cnn

from functools import partial
import numpy as np

mini = True


def load_model():
    model_fn = partial(
        gaii_joint_cnn.fit_q,
        use_time_invariant_term=True,
        length=3,
    )
    return model_fn


def load_images():
    return np.load("data/sync_mnist.npz")


def save_and_visualize_model(model, model_dir):
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model["G"].state_dict(), model_dir / "generator.pth")
    torch.save(model["D"].state_dict(), model_dir / "discriminator.pth")
    visualize.visualize_model(model, model_dir)


def experiment_gaii():
    n_step = 500 if mini else 10000
    # GAIIの算出
    model_fn = load_model()
    images = load_images()
    result = model_fn(images["images1"], images["images2"], n_step=n_step, debug=True)
    # 学習結果の可視化・保存
    save_and_visualize_model(
        model=result,
        model_dir=Path("data/gaii_cnn1"),
    )

    result = model_fn(images["images1"], images["images3"], n_step=n_step, debug=True)

    # 学習結果の可視化・保存
    save_and_visualize_model(
        model=result,
        model_dir=Path("data/gaii_cnn2"),
    )


if __name__ == "__main__":
    fire.Fire(experiment_gaii)
