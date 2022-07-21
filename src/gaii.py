from functools import partial
import gaii_joint_dense
import gaii_joint_linear
import gaii_joint_lstm
import gaii_cond_dense
import gaii_cond_linear
import gaii_cond_lstm
import utility
import fire
import json

from functools import partial

from geoii import fit_q_reestimate
import test_data
import visualize
from pathlib import Path


def load_model(params):
    model_name = params["model_name"]
    modules = {
        "gaii_joint_dense": gaii_joint_dense,
        "gaii_joint_linear": gaii_joint_linear,
        "gaii_joint_lstm": gaii_joint_lstm,
        "gaii_cond_dense": gaii_cond_dense,
        "gaii_cond_linear": gaii_cond_linear,
        "gaii_cond_lstm": gaii_cond_lstm,
    }
    model_fn = partial(
        modules[model_name].fit_q,
        use_time_invariant_term=params["use_time_invariant_term"],
        length=params["length"],
    )
    return model_fn


def save_and_visualize_model(model, model_dir):
    pass
    # model_dir.mkdir(parents=True, exist_ok=True)
    # torch.save(model["G"].state_dict(), model_dir / "generator.pth")
    # torch.save(model["D"].state_dict(), model_dir / "discriminator.pth")
    # new_process(visualize.visualize_model)(model, model_dir)


def save_result(result_all, candidate_list, data_dir):
    pass
    # result_pd = pd.DataFrame.from_records(result_all, index=candidate_list)
    # data_dir.mkdir(parents=True, exist_ok=True)
    # new_process(visualize.plot_result_all)(result_pd, data_dir)


def save_entropy(param, entropy):
    with Path(f"{param['experiment_dir']}/entropy.txt").open("w") as f:
        f.write(
            f"{param['data_name']}\t{param['partation']}\t{param['model_name']}\t{entropy}\n"
        )


def experiment_gaii(param):
    print(f"data: {param['data_name']}")
    print(f"partation: {param['partation']}")
    print(f"model: {param['model_name']}")

    # GAIIの算出
    model_fn = load_model(param)
    state_list = test_data.load_state_list(param)
    partation = param["partation"]
    result = model_fn(state_list, partation, debug=param["debug"], n_step=100)

    # 学習結果の可視化・保存
    save_and_visualize_model(
        model=result,
        model_dir=param["experiment_dir"],
    )
    save_entropy(param, result["js"])


def experiment_geometric(param):
    print(f"data: {param['data_name']}")
    print(f"partation: {param['partation']}")
    print("model: geometric")

    state_list = test_data.load_state_list(param)
    partation = param["partation"]
    result = fit_q_reestimate(state_list, partation, debug=param["debug"])
    save_entropy(param, result["kl"])


def experiment_mi(param):
    print(f"data: {param['data_name']}")
    print(f"partation: {param['partation']}")
    print("model: mutual infomation")

    state_list = test_data.load_state_list(param)
    save_entropy(param, utility.calc_MI(state_list))


def experiment(param_filename, debug=False):
    with Path(param_filename).open() as f:
        param = json.load(f)
        print(param)

    param |= {"debug": debug}
    if param["type"] == "gaii":
        experiment_gaii(param)
    elif param["type"] == "geometric":
        experiment_geometric(param)
    else:
        experiment_mi(param)


if __name__ == "__main__":
    fire.Fire(experiment)
