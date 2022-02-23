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

from src.geoii import fit_q_reestimate
import test_data


def product_dict(d):
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


def experiment_gaii():
    result_all = []
    for partation, state_list in test_data.iterate_data():
        result = {}
        for name, model_fn in iterate_model:
            model_gaii = model_fn(state_list, partation, debug=True, n_step=20000)
            result[name] = model_gaii["js"]

        model_geoii = fit_q_reestimate(state_list, partation, debug=True)
        result["geoii"] = model_geoii["kl"]

        result["MI"] = utility.calc_MI(state_list)
        result_all.append(result)
