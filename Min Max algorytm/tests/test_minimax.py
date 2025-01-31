from connect_four_game.player import Player
from connect_four_game.connect_four import ConnectFour
from minimax import MinMaxRunner

ROW_COUNT = 6
COLUMN_COUNT = 7


def test_simple_choice_going_for_win():
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


def test_simple_choice_going_for_win2():
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


def test_tournament_simple():
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    runner = MinMaxRunner(game)

    while True:
        runner.make_minimax_move(3)
        if game.state.is_finished():
            break
        runner.make_minimax_move(1)
        if game.state.is_finished():
            break

    assert game.state.get_winner() == p1


def test_tournament():
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    runner = MinMaxRunner(game)

    while True:
        runner.make_minimax_move(5)
        if game.state.is_finished():
            break
        runner.make_minimax_move(2)
        if game.state.is_finished():
            break

    assert game.state.get_winner() == p1


def test_tournament2():
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    runner = MinMaxRunner(game)

    while True:
        runner.make_minimax_move(3)
        if game.state.is_finished():
            break
        runner.make_minimax_move(4)
        if game.state.is_finished():
            break

    assert game.state.get_winner() == p2


def test_tournament_same_depths():
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    runner = MinMaxRunner(game)

    while True:
        runner.make_minimax_move(3)
        if game.state.is_finished():
            break
        runner.make_minimax_move(3)
        if game.state.is_finished():
            break

    # Winner is the one who started
    assert game.state.get_winner() == p1


def test_blocking():
    """
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][b][ ][b][ ][ ][ ]
    [ ][b][b][a][ ][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p2, second_player=p1)
    game.state.fields = [[None, None, None, None, None, None],
                         [p2, p2, None, None, None, None],
                         [p2, None, None, None, None, None],
                         [p1, p2, p1, p1, p1, p2],
                         [None, None, None, None, None, None],
                         [p1, None, None, None, None, None],
                         [p1, None, None, None, None, None]]
    runner = MinMaxRunner(game)
    assert (p2, 4) == runner.show_minimax_move(5)[1:]
    runner.make_minimax_move(5)
    assert (p1, 2) == runner.show_minimax_move(5)[1:]


def test_blocking2():
    """
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][a][a][ ][ ][ ]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p2, second_player=p1)
    game.state.fields = [[None, None, None, None, None, None],
                         [None, None, None, None, None, None],
                         [p1, None, None, None, None, None],
                         [p1, p2, p1, p1, p1, p2],
                         [None, None, None, None, None, None],
                         [None, None, None, None, None, None],
                         [None, None, None, None, None, None]]
    runner = MinMaxRunner(game)
    assert runner.show_minimax_move(5)[1:] in [(p2, 1), (p2, 4)]


def test_blocking3():
    """
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][ ][b][ ][ ][ ]
    [ ][ ][ ][b][ ][ ][ ]
    [ ][a][b][a][ ][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p2, second_player=p1)
    game.state.fields = [[None, None, None, None, None, None],
                         [p1, None, None, None, None, None],
                         [p2, None, None, None, None, None],
                         [p1, p2, p2, p2, p1, None],
                         [None, None, None, None, None, None],
                         [p1, None, None, None, None, None],
                         [p1, None, None, None, None, None]]
    runner = MinMaxRunner(game)
    assert (p2, 4) == runner.show_minimax_move(5)[1:]


def test_first_move():
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p2, second_player=p1)
    runner = MinMaxRunner(game)
    assert (p2, 3) == runner.show_minimax_move(5)[1:]
