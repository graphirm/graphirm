"""Cash drawer balance in integer cents."""


class Drawer:
    def __init__(self) -> None:
        self.cents = 0

    def deposit(self, cents: int) -> None:
        if cents < 0:
            raise ValueError("negative deposit")
        self.cents += cents
