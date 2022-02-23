from functools import partial
import gaii_joint_dense
import gaii_joint_linear
import gaii_joint_lstm
import gaii_cond_dense
import gaii_cond_linear
import gaii_cond_lstm
import utility

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


def iterate_model():
    modules = [
        gaii_joint_dense,
        gaii_joint_linear,
        gaii_joint_lstm,
        gaii_cond_dense,
        gaii_cond_linear,
        gaii_cond_lstm,
    ]
    hyperparams = product_dict(
        modules=modules, use_time_invariant_term=[True, False], length=[4, 8, 16]
    )

    for h in hyperparams:
        model_fn = partial(
            h["modules"].fit_q,
            use_time_invariant_term=h["use_time_invariant_term"],
            length=h["length"],
        )
        yield [h["modules"].__name__, model_fn]


def create_dir():
    now = datetime.datetime.now()
    datadir = Path("data/exp_gaii_div") / now.strftime("%Y%m%d-%H%M%S")
    datadir.mkdir(parents=True, exist_ok=True)
    return datadir


def experiment_gaii():
    datadir = create_dir()
    data_names = []
    result_all = []
    for data_name, partation, state_list in test_data.iterate_data():
        print(f"data: {data_name}")
        result = {}
        for model_name, model_fn in iterate_model():
            print(f"model: {model_name}")
            localdir = datadir.joinpath(f"data={data_name}_model={model_name}")
            model_gaii = model_fn(state_list, partation, debug=True, n_step=20000)
            visualize.failure_check(model_gaii, localdir)
            visualize.js_all(model_gaii, localdir)
            visualize.loss_all(model_gaii, localdir)
            visualize.FID_all(model_gaii, localdir)
            result[model_name] = model_gaii["js"]

        model_geoii = fit_q_reestimate(state_list, partation, debug=True)
        result["geoii"] = model_geoii["kl"]

        result["MI"] = utility.calc_MI(state_list)
        result_all.append(result)
        data_names.append(data_name)
        result_pd = pd.DataFrame.from_records(result_all, index=data_names)
        localdir = datadir.joinpath(f"until={data_name}")
        result_pd.to_pickle(localdir.joinpath(f"result.pkl"))
        visualize.plot_all_result(result_pd, localdir)

    result_pd = pd.DataFrame.from_records(result_all, index=data_names)
    localdir = datadir.joinpath(f"output")
    result_pd.to_pickle(localdir.joinpath(f"result.pkl"))
    visualize.plot_all_result(result_pd, localdir)
