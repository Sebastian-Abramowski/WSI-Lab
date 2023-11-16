from typing import Tuple, List

from player import Player
from connect_four import ConnectFour, ConnectFourMove


ROW_COUNT = 6
COLUMN_COUNT = 7

alpha = float('-inf')
beta = float('inf')


p1 = Player("a")
p2 = Player("b")
game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)

game.make_move(ConnectFourMove(3))
game.make_move(ConnectFourMove(4))
game.make_move(ConnectFourMove(3))
print(game)

# print([str(x) for x in game.state.get_moves()])

print(game.state.fields)

for i, row in enumerate(game.state.fields):
    # print(row)
    pass
