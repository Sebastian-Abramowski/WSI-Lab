from typing import Tuple, List

from player import Player
from connect_four import ConnectFour, ConnectFourMove
from minimax import MinMaxSolver


ROW_COUNT = 6
COLUMN_COUNT = 7

alpha = float('-inf')
beta = float('inf')


p1 = Player("a")
p2 = Player("b")
game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)

game.make_move(ConnectFourMove(3))
game.make_move(ConnectFourMove(3))
game.make_move(ConnectFourMove(3))
game.make_move(ConnectFourMove(3))
game.make_move(ConnectFourMove(3))
game.make_move(ConnectFourMove(3))
game.make_move(ConnectFourMove(4))
game.make_move(ConnectFourMove(5))
game.make_move(ConnectFourMove(2))
game.make_move(ConnectFourMove(6))
game.make_move(ConnectFourMove(1))
game.make_move(ConnectFourMove(2))
game.make_move(ConnectFourMove(1))
game.make_move(ConnectFourMove(6))
game.make_move(ConnectFourMove(1))
game.make_move(ConnectFourMove(6))
game.make_move(ConnectFourMove(1))
game.make_move(ConnectFourMove(6))
print(game.state.fields)

# print([str(x) for x in game.state.get_moves()])

# print([x.column for x in game.get_moves()])
# print(game.state.is_finished())
print(game)
solver = MinMaxSolver(game)
solver.get_best_move(game.state)

print(solver.count_vertical(game.state, p2))
