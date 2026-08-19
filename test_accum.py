"""Self-check cho gradient accumulation. Chay: python test_accum.py

Soi hai cho de sai am tham trong vong lap train_one_epoch:
1. loss PHAI chia accum — khong chia thi gradient la tong chu khong phai trung
   binh, cong voi LR da x accum o basetrainer.make_optimizer -> lech accum^2.
2. micro-batch cuoi cua epoch chia khong het accum PHAI duoc step, khong thi
   gradient cua chung treo sang epoch sau (zero_grad khong duoc goi).
"""

import torch


def step_indices(num_steps: int, accum: int) -> list[int]:
    """Cac idx ma vong lap that su goi optimizer.step() — copy dieu kien tu
    seg_trainer.train_one_epoch / cls_trainer.train_one_epoch."""
    return [
        idx
        for idx in range(num_steps)
        if not ((idx + 1) % accum != 0 and (idx + 1) != num_steps)
    ]


def main() -> None:
    # accum=1: step moi batch, y het hanh vi cu
    assert step_indices(5, 1) == [0, 1, 2, 3, 4]
    # chia het: step o cuoi moi nhom
    assert step_indices(8, 4) == [3, 7]
    # chia khong het: batch cuoi (idx=8) van phai step
    assert step_indices(9, 4) == [3, 7, 8]
    assert step_indices(3, 4) == [2]

    # Gradient tich luy voi loss/accum == gradient cua mot batch lon
    accum, feat = 4, 3
    torch.manual_seed(0)
    x = torch.randn(accum * 2, feat)
    y = torch.randn(accum * 2, 1)

    w_full = torch.zeros(feat, 1, requires_grad=True)
    torch.nn.functional.mse_loss(x @ w_full, y).backward()

    w_accum = torch.zeros(feat, 1, requires_grad=True)
    for i in range(accum):
        xb, yb = x[i * 2 : i * 2 + 2], y[i * 2 : i * 2 + 2]
        (torch.nn.functional.mse_loss(xb @ w_accum, yb) / accum).backward()

    assert torch.allclose(w_full.grad, w_accum.grad, atol=1e-6), (
        f"lech: {w_full.grad.flatten()} vs {w_accum.grad.flatten()}"
    )

    print("ok")


if __name__ == "__main__":
    main()
