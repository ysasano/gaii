import datetime
import itertools
import json
from pathlib import Path
import fire

import mip


def create_experiment_dir(now, name):
    experiment_dir = Path("data/exp_gaii_div") / now / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return str(experiment_dir)


def create_state_list_param():
    data_list = [
        ("VAR(1, 1D)", 1),
        ("VAR(1, 2D)", 2),
        ("VAR(1, 3D)", 3),
        ("VAR(4, 1D)", 1),
        ("VAR(4, 2D)", 2),
        ("VAR(4, 3D)", 3),
        ("NLVAR(3, 3D)", 3),
        ("HENON", 3),
        ("LORENZ", 3),
    ]

    for (st1, d1), (st2, d2) in itertools.combinations(data_list, 2):
        param = {}
        param["state_list1"] = st1
        param["state_list2"] = st2
        param["data_name"] = f"{st1}-{st2}"
        dim = d1 + d2
        for partation in mip.generate_candidate_list(dim):
            param["partation"] = partation
            yield param


def create_model_list_param():
    model_list = [
        "gaii_joint_dense",
        "gaii_joint_linear",
        "gaii_joint_lstm",
        "gaii_cond_dense",
        "gaii_cond_linear",
        "gaii_cond_lstm",
    ]
    use_time_invariant_term = [True, False]
    length = [2, 4, 8]
    for m, t, l in itertools.product(model_list, use_time_invariant_term, length):
        param = {}
        param["model_name"] = m
        param["use_time_invariant_term"] = t
        param["length"] = l
        yield param


def create_param():
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # GAIIのパラメータ生成
    state_list_param = list(create_state_list_param())
    model_list_param = list(create_model_list_param())
    for st_param, m_param in itertools.product(state_list_param, model_list_param):
        name = f"data={st_param['data_name']}_partation={st_param['partation']}_model={m_param['model_name']}"
        param = {
            "type": "gaii",
            "experiment_dir": create_experiment_dir(now, name),
        }
        yield param | st_param | m_param

    # 幾何的統合情報量のパラメータ生成
    for st_param in state_list_param:
        name = f"data={st_param['data_name']}_partation={st_param['partation']}_model=geometric"
        param = {
            "type": "geometric",
            "model_name": "geometric",
            "experiment_dir": create_experiment_dir(now, name),
        }
        yield param | st_param

    # 相互情報量のパラメータ生成
    for st_param in state_list_param:
        name = (
            f"data={st_param['data_name']}_partation={st_param['partation']}_model=mi"
        )
        param = {
            "type": "mi",
            "model_name": "mi",
            "experiment_dir": create_experiment_dir(now, name),
        }
        yield param | st_param


def create_param_file():
    for param in create_param():
        with Path(f"{param['experiment_dir']}/setting.json").open("w") as f:
            json.dump(param, f, indent=4)


if __name__ == "__main__":
    fire.Fire(create_param_file)
