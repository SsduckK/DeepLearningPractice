import numpy as np


class SingleLayer:
    def __init__(self) -> None:
        self.weight = np.ones([2, 2], dtype=np.float32)
        self.bias = np.ones([2], dtype=np.float32)
        self.x_cache = None

    def forward(self, x):
        self.x_cache = x
        return self.weight @ x + self.bias


def main():
    input_sample = np.array([19, -96])
    GT = np.array([6, -19])

    layer = SingleLayer()

    output = layer.forward(input_sample)

    print(output)

    print(np.abs(output - GT))


if __name__ == "__main__":
    main()
