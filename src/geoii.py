import torch
import torch.optim as optim
from numpy.linalg import det
import numpy as np
from utility import to_torch, from_torch
from statsmodels.tsa.api import VAR
import mip


def fit_q_reestimate(state_list, partation, debug=False):
    var_model = VAR(state_list).fit(trend="n", maxlags=1)
    mask = mip.partation_to_mask(partation)
    return fit_q(var_model.params, var_model.sigma_u, mask, debug=debug)


def fit_q(a, sigma_e, sigma_x, mask=np.array([[1, 0], [0, 1]]), debug=False):
    A = to_torch(a)
    Mask = to_torch(mask)
    SigmaE = to_torch(sigma_e)
    SigmaX = to_torch(sigma_x)

    # 最適化
    losses_log = []
    losses_retry = []
    phig_log = []
    phig_retry = []
    loss_min = None
    # 初期値探索
    for retry in range(10):
        # 初期値を変えてループ
        baseA_ = torch.normal(
            mean=0, std=np.sqrt(1 / 2), size=a.shape, requires_grad=True
        )
        # Lossの設定
        optimizer = optim.LBFGS([baseA_], lr=0.05)
        # optimizer = optim.SGD([baseA_],lr=0.01)

        losses = []
        phig_list = []
        for _ in range(40):

            def closure():
                optimizer.zero_grad()
                # 対角成分をゼロにする
                A_ = baseA_ * Mask
                # (3)を(2)に代入
                dA = (A - A_).detach()  # ここまでバックプロパゲーションすると解が増えるためデタッチ
                SigmaE_ = SigmaE + dA @ SigmaX @ torch.t(dA)  # ...(2)
                rhs = Mask * (SigmaX @ (A - A_) @ torch.inverse(SigmaE_))  # ...(3)の右辺
                # lossを計算
                loss = torch.sum(rhs ** 2) + 0.0001 * torch.sum(
                    baseA_ ** 2
                )  # (3)式 ＝ 0 ＋ 正則化
                loss.backward()
                return loss

            l = optimizer.step(closure)
            loss_value = from_torch(l)
            losses.append(loss_value)

            # if loss_value < 0.1 or loss_value > 3:
            #    break

            A_ = baseA_ * Mask
            dA = A - A_
            SigmaE_ = SigmaE + dA @ SigmaX @ torch.t(dA)
            SigmaE_val = from_torch(SigmaE_)
            phi_G = 1 / 2 * np.log(det(SigmaE_val) / det(sigma_e))
            phig_list.append(phi_G)

        if debug:
            print("==result retry={}==".format(retry))
            print("loss=", loss_value)
            print("A=", from_torch(baseA_))

        if not phig_retry or phi_G < min(phig_retry):
            A_ = baseA_ * Mask
            A_min = from_torch(A_)
            dA = A - A_
            SigmaE_ = SigmaE + dA @ SigmaX @ torch.t(dA)
            SigmaE_min = from_torch(SigmaE_)
            loss_min = loss_value

        losses_retry.append(loss_value)
        losses_log.append(losses)
        phig_retry.append(phi_G)
        phig_log.append(phig_list)

    return {
        "A_min": A_min,
        "SigmaE_min": SigmaE_min,
        "loss_min": loss_min,
        "loss_log": losses_log,
        "phig_log": phig_log,
        "kl": 1 / 2 * np.log(det(SigmaE_min) / det(sigma_e)),
    }
