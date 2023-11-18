from __future__ import annotations


class Player:
    def __init__(self, char: str) -> None:
        """
        Initializes a player.

        Parameters:
            char: a single-character string to represent the player in textual representations of game state
        """
        if len(char) != 1:
            raise ValueError('Character that represents player should be of length 1')

        self.char = char

    def __eq__(self, other: Player) -> bool:
        if not other:
            return False
        return self.char == other.char

    def __hash__(self) -> int:
        return hash(self.char)
