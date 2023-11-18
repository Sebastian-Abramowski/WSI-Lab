from connect_four_game.player import Player
from connect_four_game.connect_four import ConnectFour
from minimax import MinMaxRunner
import tests.process_times as process_times


p1 = Player("a")
p2 = Player("b")
game = ConnectFour(size=(6, 7), first_player=p2, second_player=p1)
game.state.fields = [[None, None, None, None, None, None],
                     [p2, p2, p1, None, None, None],
                     [p2, p1, p2, None, None, None],
                     [p1, p2, p1, p2, p1, p2],
                     [p2, None, None, None, None, None],
                     [p2, p1, None, None, None, None],
                     [p1, None, None, None, None, None]]
runner = MinMaxRunner(game)

print(process_times.get_time(runner.show_minimax_move, 6))

"""
For depth 6:
time of minimax with prunning ~ 0.046875
time of minimax without prunning ~ 6.03125
"""
