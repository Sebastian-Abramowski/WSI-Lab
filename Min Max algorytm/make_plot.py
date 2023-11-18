import random

import plot
from connect_four_game.player import Player
from connect_four_game.connect_four import ConnectFour
from minimax import MinMaxRunner

ROW_COUNT = 6
COLUMN_COUNT = 7


def check_minimax(number: int, depth: int) -> list[int, int, int]:
    wins = 0
    loses = 0
    draws = 0
    for _ in range(number):
        p1 = Player("a")
        p2 = Player("b")
        if random.randint(1, 2) == 1:
            game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
        else:
            game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p2, second_player=p1)
        runner = MinMaxRunner(game)

        while not game.is_finished():
            if game.state.get_current_player() == p1:
                runner.make_minimax_move(depth)
                continue

            if game.state.get_current_player() == p2:
                moves = game.get_moves()
                move = random.choice(moves)
                game.make_move(move)
                continue

        if game.state.get_winner() == p1:
            wins += 1
        elif game.state.get_winner() == p2:
            loses += 1
        else:
            draws += 1
    return wins, loses, draws


if __name__ == "__main__":
    wins, loses, draws = check_minimax(40, 2)
    plot.configurate_plot("Results", "Number of wins", "Minimax depth 2 vs random player",
                          window_title="Minimax")
    plot.make_bar_plot(["wins", "loses", "draws"], [wins, loses, draws],
                       color=plot.PLOT_CYAN)
    plot.show_plot()
