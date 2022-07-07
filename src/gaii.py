from functools import partial
import gaii_joint_dense
import gaii_joint_linear
import gaii_joint_lstm
import gaii_cond_dense
import gaii_cond_linear
import gaii_cond_lstm
from jinja2 import pass_context
import utility
import mip
import fire
import torch
import multiprocessing

from functools import partial
from itertools import product

from geoii import fit_q_reestimate
import test_data
import visualize
import datetime
from pathlib import Path
import pandas as pd


def product_dict(**d):
    prod = [x for x in product(*d.values())]
    return (dict(zip(d.keys(), v)) for v in prod)


def generate_model_list():
    modules = [
        gaii_joint_dense,
        gaii_joint_linear,
        gaii_joint_lstm,
        gaii_cond_dense,
        gaii_cond_linear,
        gaii_cond_lstm,
    ]
    hyperparams = product_dict(
        modules=modules, use_time_invariant_term=[True, False], length=[2, 4, 8]
    )

    result = []
    for h in hyperparams:
        model_fn = partial(
            h["modules"].fit_q,
            use_time_invariant_term=h["use_time_invariant_term"],
            length=h["length"],
        )
        name = h["modules"].__name__
        name += "_time_invariant=" + str(h["use_time_invariant_term"])
        name += "_length=" + str(h["length"])
        result.append([name, model_fn])
    return result


def create_experiment_dir():
    now = datetime.datetime.now()
    experiment_dir = Path("data/exp_gaii_div") / now.strftime("%Y%m%d-%H%M%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


# Matplotlibの不具合回避策
# MacOSXでsavefigするとメモリリークする問題(※)への対応
# [Bug]: macosx timers don't fire if plt.show() hasn't been called
#    https://github.com/matplotlib/matplotlib/issues/23061
# FIX: Decrease figure refcount on close of a macosx figure #23059
#    https://github.com/matplotlib/matplotlib/pull/23059
#
# 解決策: 描画処理を別プロセスに隔離するように変更
# 参考にしたページ: https://qiita.com/Masahiro_T/items/037c118c9dd26450f8c8
def new_process(func):
    def wrapper(*args, **kw):
        p = multiprocessing.Pool(1)
        p.apply(func, args, kw)
        p.close()

    return wrapper


def save_and_visualize_model(model, model_dir):
    pass
    #model_dir.mkdir(parents=True, exist_ok=True)
    #torch.save(model["G"].state_dict(), model_dir / "generator.pth")
    #torch.save(model["D"].state_dict(), model_dir / "discriminator.pth")
    # new_process(visualize.visualize_model)(model, model_dir)


def save_result(result_all, candidate_list, data_dir):
    pass
    #result_pd = pd.DataFrame.from_records(result_all, index=candidate_list)
    #data_dir.mkdir(parents=True, exist_ok=True)
    # new_process(visualize.plot_result_all)(result_pd, data_dir)


def experiment_gaii(trial_mode=False, debug=False):
    # 実験用データディレクトリを作成
    experiment_dir = create_experiment_dir()

    # データとモデルを作成 (分割はデータに依存するためここでは作成しない)
    data_list = test_data.generate_data_list()
    model_list = generate_model_list()

    # (データ, 分割, モデル)の三重ループ
    for data_idx, (data_name, dim, state_list) in enumerate(data_list):
        result_by_data = []

        # 分割を作成
        candidate_list = mip.generate_candidate_list(dim)
        for candidate_idx, partation in enumerate(candidate_list):
            result = {}
            for model_idx, (model_name, model_fn) in enumerate(model_list):
                # ループのインデックスを出力
                print(f"data: {data_name} ({data_idx}/{len(data_list)})")
                print(f"partation: {partation} ({candidate_idx}/{len(candidate_list)})")
                print(f"model: {model_name} ({model_idx}/{len(model_list)})")

                # GAIIの算出
                model_gaii = model_fn(state_list, partation, debug=debug, n_step=20000)
                result[model_name] = model_gaii["js"]

                # 学習結果の可視化・保存
                save_and_visualize_model(
                    model=model_gaii,
                    model_dir=experiment_dir
                    / f"data={data_name}_partation={partation}_model={model_name}",
                )
                if trial_mode:
                    break

            # 幾何的統合情報量の算出
            model_geoii = fit_q_reestimate(state_list, partation, debug=debug)
            result["geoii"] = model_geoii["kl"]

            # 相互情報量の算出
            result["MI"] = utility.calc_MI(state_list)

            # データごとの結果のリストを更新
            result_by_data.append(result)

            # データごとの結果のリストを保存
            save_result(
                result_all=result_by_data,
                candidate_list=candidate_list[: len(result_by_data)],
                data_dir=experiment_dir / f"data={data_name}",
            )
            if trial_mode:
                break

        # データごとの結果のリストを保存
        save_result(
            result_all=result_by_data,
            candidate_list=candidate_list,
            data_dir=experiment_dir / f"data={data_name}",
        )
        if trial_mode:
            break


if __name__ == "__main__":
    fire.Fire(experiment_gaii)
