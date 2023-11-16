from __future__ import annotations
from typing import Iterable, Optional

from move import Move
from player import Player


class State:
    """Immutable game state object"""
    def __init__(self, current_player, other_player) -> None:
        self._current_player = current_player
        self._other_player = other_player

    def get_moves(self) -> Iterable[State]:
        """
        Returns:
            Possible moves
        """
        raise NotImplementedError

    def get_current_player(self) -> Player:
        """
        Returns:
            Current player
        """
        return self._current_player

    def make_move(self, move: Move) -> State:
        """
        Creates a new state after making the move

        Parameters:
            move: the move to make

        Returns:
            The state after the move
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """
        Returns:
            If the game is finished
        """
        raise NotImplementedError

    def get_winner(self) -> Optional[Player]:
        """
        Returns:
            The player that won or None if draw or not finished
        """
        raise NotImplementedError

    def get_players(self) -> Iterable[Player]:
        """
        Return:
            Iterable of the players in the game
        """
        return [self._current_player, self._other_player]

    def __str__(self) -> str:
        """
        Returns:
            The string representation of the game's state
        """
        raise NotImplementedError
