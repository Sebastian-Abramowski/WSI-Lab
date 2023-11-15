from connect_four import ConnectFour
import random


game = ConnectFour()

while not game.is_finished():
    moves = game.get_moves()
    move = random.choice(moves)
    game.make_move(move)
    print(game.state)

winner = game.get_winner()
if winner is None:
    print('Draw!')
else:
    print('Winner: Player ' + winner.char)
