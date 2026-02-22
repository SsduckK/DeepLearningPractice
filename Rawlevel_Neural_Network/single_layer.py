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
        x = self.x_cache
        dW = np.outer(dL_dout, x)
        db = dL_dout
        dx = self.weight.T @ dL_dout

        return dW, db, dx

    def step(self, dw, db, lr):
        self.weight -= lr * dw
        self.bias -= lr * db


def MSE_loss(pred, gt):
    diff = pred - gt
    loss = 0.5 * np.sum(diff**2)
    dL_pred = diff
    return loss, dL_pred


def main():
    x = np.array([19, -96])
    GT = np.array([6, -19])

    layer = SingleLayer()
    lr = 1e-4

    for i in range(1, 10):
        pred = layer.forward(x)
        loss, dL_pred = MSE_loss(pred, GT)

        dW, db, _ = layer.backward(dL_pred)
        layer.step(dW, db, lr)

        print(f"i={i} loss:{loss:.4f} pred{pred}")


if __name__ == "__main__":
    main()
