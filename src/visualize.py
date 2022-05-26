import numpy as np
from utility import to_torch, from_torch, tril
import matplotlib.pyplot as plt
import gaii_cond_linear
import torch


def imshow(model, datadir=None):  # show_matrix
    _, axes = plt.subplots(2, 1)
    w = from_torch(model["G"].state_dict()["fc1.weight"] * model["mask"])
    print("A = \n", w)
    axes[0].imshow(w)

    N = model["N"]
    L = from_torch(model["G"].state_dict()["fc2.weight"]) * from_torch(tril(N))
    print("Σ(E) = \n", L @ L.T)
    axes[1].imshow(L @ L.T)
    if datadir:
        plt.savefig(datadir.joinpath("imshow.png"))


def simulate(model, datadir=None):  # show_simulate
    real_x, real_y = gaii_cond_linear.sample_xy(1, model["state_list"])
    _, axes = plt.subplots(2, 1)

    def plot(ax, j):
        fake_ys = []
        true_ys = []
        for _ in range(2000):
            z = gaii_cond_linear.sample_z(1, model["N"])
            fake_y = model["G"](real_x, z)
            fake_ys.append(from_torch(fake_y)[0, j])
            E = np.random.multivariate_normal(
                np.zeros(initiali_state.shape), Sigma_true, size=1
            )[0]
            true_y = list(A @ np.array(real_x[0]) + E.T)
            true_ys.append(true_y[j])

        print(fake_ys)
        print("x", real_x[0, 0])
        print("y", real_y[0, 0])
        print(true_ys)
        # ax.axis("off")
        plt.figure()
        ax.hist(fake_ys, 100, alpha=0.3)
        ax.hist(
            true_ys,
            100,
            alpha=0.3,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
        )
        if datadir:
            plt.savefig(datadir.joinpath("simulate.png"))

    plot(axes[0], 0)
    plot(axes[1], 1)


def mode_check(model, datadir=None):  # show_mode_collapse
    xs = to_torch(np.tile(np.arange(-2.0, 2.0, 0.01), (model["N"], 1))).T
    zs = gaii_cond_linear.sample_z(len(xs), model["N"])
    # print(xs.shape, zs.shape)
    ys = model["G"](xs, zs)
    # print(xs)
    plt.figure()
    plt.plot(from_torch(xs)[:, 0], from_torch(ys)[:, 0])
    if datadir:
        plt.savefig(datadir.joinpath("mode_check.png"))


def failure_check(model, datadir=None):
    plt.figure()
    model["failure_check"].plot(ylim=[0, 1])
    if datadir:
        model["failure_check"].to_pickle(datadir.joinpath(f"failure_check.pkl"))
        plt.savefig(datadir.joinpath("failure_check.png"))


def js_all(model, datadir=None):
    plt.figure()
    model["js_all"].plot()
    if datadir:
        model["js_all"].to_pickle(datadir.joinpath(f"js.pkl"))
        plt.savefig(datadir.joinpath("js.png"))


def FID_all(model, datadir=None):
    plt.figure()
    model["FID_all"].plot()
    if datadir:
        model["FID_all"].to_pickle(datadir.joinpath(f"FID.pkl"))
        plt.savefig(datadir.joinpath("FID.png"))


def loss_all(model, datadir=None):
    plt.figure()
    model["loss_all"].plot()
    if datadir:
        model["loss_all"].to_pickle(datadir.joinpath(f"loss.pkl"))
        plt.savefig(datadir.joinpath("loss.png"))

def plot_result_all(pd_result, datadir=None):
    # 全件をplot
    plot_result(pd_result, datadir, "all")

    # GeoiiとMIを除外したGAIIのみでplot
    gaii_only = [col for col in pd_result.columns if "gaii" in col]
    plot_result(pd_result[gaii_only], datadir, "gaii_only")

    # 要素ごとにplot
    for col in pd_result.columns:
        plot_result(pd_result[col], datadir, col)


def plot_result(pd_result, datadir=None, suffix=""):
    plt.figure()
    ax = pd_result.plot()
    ax.tick_params(axis="x", rotation=70)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    if datadir:
        pd_result.to_pickle(datadir.joinpath(f"result.pkl"))
        plt.savefig(datadir.joinpath(f"result_{suffix}.png"), bbox_inches="tight")
