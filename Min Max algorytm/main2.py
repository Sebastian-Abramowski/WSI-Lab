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
game.make_move(ConnectFourMove(5))
game.make_move(ConnectFourMove(2))
game.make_move(ConnectFourMove(6))
game.make_move(ConnectFourMove(1))
game.make_move(ConnectFourMove(2))
game.make_move(ConnectFourMove(1))
game.make_move(ConnectFourMove(2))
game.make_move(ConnectFourMove(1))
game.make_move(ConnectFourMove(5))
# game.make_move(ConnectFourMove(4))
print(game)

# print([str(x) for x in game.state.get_moves()])

# print([x.column for x in game.get_moves()])
# print(game.state.is_finished())
solver = MinMaxSolver()
print(solver.count_horizontal(game.state, p1))
# print(game.state.fields)
# print(solver.evaluate_for_player(game.state, game.state._other_player))
# print(solver.minimax(game.state, 5, alpha, beta, True)[1])

# TODO: porównanie czasu wykonywania z alpha beta prunnigiem i bez dla jednego przypadku w środku rozgrywki
# TODO: spójrz sobie na kod, ogarnij czy są jakieś powtórzenia, ostatnie szlify, zrób metody prywatne, za dużo wcięć potrzebny refactor
# TODO: prrównanie głębokość 5 vs głębokość 2, kto ile razy wygrywa, przeprowadź 50 testów
# TODO: napisać notatke o Min maxie
# TODO: wyczyścić strukturę folderu (usunąc niepotrzebne pliki, linter itp.)
# TODO: sprawozdanie, przeklejenie kodu, jakieś wnioski
# TODO: trzeba cały folder wrzucić do Google Colaba
