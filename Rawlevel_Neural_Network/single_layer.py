import numpy as np


class SingleLayer:
    def __init__(self) -> None:
        self.weight = np.ones([2, 2])
        self.bias = np.ones([2])


def main():
    input_sample = np.array([19, -96])
    GT = np.array([6, -19])

    layer = SingleLayer()


if __name__ == "__main__":
    main()
