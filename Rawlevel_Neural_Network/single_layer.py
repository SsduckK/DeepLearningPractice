import numpy as np


class SingleLayer:
    def __init__(self) -> None:
        self.weight = np.ones([2, 2], dtype=np.float32)
        self.bias = np.ones([2], dtype=np.float32)
        self.x_cache = None

    def forward(self, x):
        self.x_cache = x
        return self.weight @ x + self.bias

    def backward(self, dL_dout):
        pass

    def step(self, dw, db, lr):
        pass


def MSE_loss(pred, gt):
    diff = pred - gt
    loss = 0.5 * np.sum(diff**2)
    dL_pred = diff
    return loss, dL_pred


def main():
    input_sample = np.array([19, -96])
    GT = np.array([6, -19])

    layer = SingleLayer()

    pred = layer.forward(input_sample)

    loss, dL_pred = MSE_loss(pred, GT)
    print(loss, dL_pred)


if __name__ == "__main__":
    main()
