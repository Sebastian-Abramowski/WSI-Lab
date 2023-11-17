from player import Player
from connect_four import ConnectFour, ConnectFourMove
from minimax import MinMaxRunner

ROW_COUNT = 6
COLUMN_COUNT = 7


def test_simple_choice():
    """
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][b][a][b][ ][ ][ ]
    [ ][b][b][a][ ][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    game.state.fields = [[None, None, None, None, None, None],
                         [p2, p2, None, None, None, None],
                         [p2, p1, None, None, None, None],
                         [p1, p2, p1, p2, p1, p2],
                         [None, None, None, None, None, None],
                         [p1, None, None, None, None, None],
                         [p1, None, None, None, None, None]]
    runner = MinMaxRunner(game)
    assert (p1, 4) == runner.show_minimax_move(1)[1:]
    assert (p1, 4) == runner.show_minimax_move(2)[1:]
    assert (p1, 4) == runner.show_minimax_move(3)[1:]
    assert (p1, 4) == runner.show_minimax_move(4)[1:]
    assert (p1, 4) == runner.show_minimax_move(5)[1:]


def test_simple_choice2():
    """
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][b][ ][ ][ ]
    [ ][b][a][a][ ][ ][ ]
    [ ][b][a][b][ ][a][ ]
    [ ][b][b][a][ ][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p2, second_player=p1)
    game.state.fields = [[None, None, None, None, None, None],
                         [p2, p2, p2, None, None, None],
                         [p2, p1, p1, None, None, None],
                         [p1, p2, p1, p2, p1, p2],
                         [None, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    runner = MinMaxRunner(game)
    assert (p2, 1) == runner.show_minimax_move(1)[1:]
    assert (p2, 1) == runner.show_minimax_move(2)[1:]
    assert (p2, 1) == runner.show_minimax_move(3)[1:]
    assert (p2, 1) == runner.show_minimax_move(4)[1:]
    assert (p2, 1) == runner.show_minimax_move(5)[1:]
