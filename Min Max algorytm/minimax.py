from typing import Tuple, Optional
import sys

from connect_four import ConnectFour, ConnectFourState
from player import Player

sys.setrecursionlimit(5000)


class MinMaxSolver:
    def evaluate_for_player(self, state: ConnectFourState, player: Player) -> float:
        if state.get_winner() == player:
            return 99999

        score = 0
        # Vertically
        for seq_length, num_of_seqs in self.count_vertical(state, player).items():
            score += 2 * seq_length**2 * num_of_seqs

        # Horizontally
        for seq_length, num_of_seqs in self.count_horizontal(state, player).items():
            score += 2 * seq_length**2 * num_of_seqs

        # Booster
        score += self.count_in_center_column(state, player)

        return score

    def count_horizontal(self, state: ConnectFourState, player: Player) -> dict[int, int]:
        counter = {1: 0, 2: 0, 3: 0}

        num_rows = len(state.fields[0])
        num_columns = len(state.fields)
        for i in range(num_rows):
            row = [state.fields[j][i] for j in range(num_columns)]
            for row_index, field in enumerate(row):
                if field == player:
                    counted = self._count_horizontal(row, player, row_index)
                    is_none_on_left = row_index - 1 >= 0 and row[row_index - 1] is None
                    is_none_on_right = row_index + counted < len(row) - 1 and row[row_index + counted] is None
                    if counted in counter and (is_none_on_left or is_none_on_right):
                        counter[counted] += 1
        return counter

    def _count_horizontal(self, row: list[Optional[Player]], player: Player, start_index: int) -> int:
        counted = 0
        for i in range(start_index, len(row)):
            if row[i] == player:
                counted += 1
                continue
            else:
                break
        return counted

    def count_in_center_column(self, state: ConnectFourState, player: Player) -> int:
        center_column_index = len(state.fields) // 2
        counted_occur_in_center = sum([1 for field in state.fields[
            center_column_index] if field == player])

        if len(state.fields) % 2 == 0:
            counted_occur_in_center += sum([1 for field in state.fields[
                center_column_index - 1] if field == player])

        return counted_occur_in_center

    def count_vertical(self, state: ConnectFourState, player: Player) -> dict[int, int]:
        counter = {1: 0, 2: 0, 3: 0}
        for i, column in enumerate(state.fields):
            for j, field in enumerate(column):
                if field is None:
                    counted = self._count_vertical(player, column, j - 1)
                    if counted in counter:
                        counter[counted] += 1
        return counter

    def _count_vertical(self, player: Player, column: list[Optional[Player]], index) -> int:
        counted = 0
        while (index >= 0):
            if column[index] == player:
                counted += 1
                index -= 1
            else:
                break
        return counted

    def minimax(self, state: ConnectFourState, depth: int, alpha: float, beta: float,
                if_max_player: bool) -> Tuple[int, ConnectFourState]:
        """Returns column index and state"""
        if depth == 0 or state.is_finished():
            eval = self.evaluate_for_player(state, state._other_player
                                            ) - self.evaluate_for_player(state, state._current_player)
            return eval, state

        if if_max_player:
            max_eval = float('-inf')
            best_state = None
            for move in state.get_moves():
                new_state = state.make_move(move)
                evaluation = self.minimax(new_state, depth - 1, alpha, beta, False)[0]
                if evaluation > max_eval:
                    max_eval = evaluation
                    best_state = new_state
                alpha = max(alpha, max_eval)
                if alpha >= beta:
                    break
            return max_eval, best_state
        else:
            min_eval = float('inf')
            best_state = None
            for move in state.get_moves():
                new_state = state.make_move(move)
                evaluation = self.minimax(new_state, depth - 1, alpha, beta, True)[0]
                if evaluation < min_eval:
                    min_eval = evaluation
                    best_state = new_state
                beta = min(beta, min_eval)
                if alpha >= beta:
                    break
            return min_eval, best_state


class MinMaxRunner():
    def __init__(self, game: ConnectFour):
        self.game = game
        self.MinMaxSolver = MinMaxSolver()

    def _get_best_move(self, best_next_state: ConnectFourState) -> int:
        """Returns column index"""
        for i, column in enumerate(best_next_state.fields):
            none_num = sum([1 for field in column if field is None])
            current_none_num = sum([1 for field in self.game.state.fields[i] if field is None])
            if none_num != current_none_num:
                return i

    def show_minimax_move(self, depth: int) -> Tuple[ConnectFourState, Player, int]:
        alpha = float('-inf')
        beta = float('inf')
        move_made_by = self.game.state.get_current_player()
        new_state = self.MinMaxSolver.minimax(self.game.state, depth, alpha, beta, True)[1]
        best_column_move = self._get_best_move(new_state)
        return new_state, move_made_by, best_column_move

    def make_minimax_move(self, depth: int) -> None:
        self.game.state = self.show_minimax_move(depth)[0]
