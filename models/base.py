"""Base model interface for training and evaluation."""


class ModelBase:
    def train(self) -> None:
        raise NotImplementedError
