from player import Player
from connect_four import ConnectFour, ConnectFourMove
from minimax import MinMaxSolver

ROW_COUNT = 6
COLUMN_COUNT = 7


def test_count_at_center():
    """
    [ ][ ][ ][ ][ ][ ][ ]
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
                         [p1, p2, p1, p2, p1, None],
                         [None, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    solver = MinMaxSolver()
    assert solver.count_in_center_column(game.state, p1) == 3
    assert solver.count_in_center_column(game.state, p2) == 2


def test_count_horizontal():
    """
    [ ][ ][ ][ ][ ][ ][ ]
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
                         [p1, p2, p1, p2, p1, None],
                         [None, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    solver = MinMaxSolver()
    assert solver.count_horizontal(game.state, p1) == {1: 3, 2: 2, 3: 0}
    assert solver.count_horizontal(game.state, p2) == {1: 4, 2: 1, 3: 0}


def test_count_vertical():
    """
    [ ][ ][ ][ ][ ][ ][ ]
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
                         [p1, p2, p1, p2, p1, None],
                         [None, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    solver = MinMaxSolver()
    assert solver.count_vertical(game.state, p1) == {1: 2, 2: 2, 3: 0}
    assert solver.count_vertical(game.state, p2) == {1: 0, 2: 0, 3: 1}


def test_nearly_win_situation_indication():
    """
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][ ][ ][b][ ][ ][ ]
    [ ][a][a][a][ ][ ][ ]
    [ ][b][a][b][ ][a][ ]
    [ ][b][b][a][ ][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p2, second_player=p1)
    game.state.fields = [[None, None, None, None, None, None],
                         [p2, p2, p1, None, None, None],
                         [p2, p1, p1, None, None, None],
                         [p1, p2, p1, p2, p1, None],
                         [None, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    solver = MinMaxSolver()
    solver.count_horizontal(game.state, p1)
    assert solver.if_potencial_winner


def test_count_diagonal_NW_SE_none_top_left():
    """
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][a][ ][ ][ ]
    [ ][a][ ][b][ ][ ][ ]
    [ ][a][a][a][ ][ ][ ]
    [ ][b][a][a][ ][a][ ]
    [ ][b][b][a][b][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    game.state.fields = [[None, None, None, None, None, None],
                         [p2, p2, p1, p1, None, None],
                         [p2, p1, p1, None, None, None],
                         [p1, p1, p1, p2, p1, None],
                         [p2, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    solver = MinMaxSolver()
    assert solver.count_diagonal_NW_SE(game.state, p1) == 2


def test_count_diagonal_NW_SE_none_right_bottom():
    """
    [ ][ ][ ][ ][ ][ ][ ]
    [b][ ][ ][ ][ ][ ][ ]
    [a][a][ ][ ][ ][ ][ ]
    [a][a][a][ ][ ][ ][ ]
    [b][b][a][ ][ ][a][ ]
    [a][b][b][ ][b][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    game.state.fields = [[p1, p2, p1, p1, p2, None],
                         [p2, p2, p1, p1, None, None],
                         [p2, p1, p1, None, None, None],
                         [None, None, None, None, None, None],
                         [p2, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    solver = MinMaxSolver()
    assert solver.count_diagonal_NW_SE(game.state, p1) == 1


def test_count_diagonal_NW_SE_double_case():
    """
    Characteristic case
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][ ][ ][ ][ ]
    [a][a][ ][ ][ ][ ][ ]
    [a][b][a][ ][ ][ ][ ]
    [b][b][a][a][ ][a][ ]
    [a][b][b][b][ ][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    game.state.fields = [[p1, p2, p1, p1, None, None],
                         [p2, p2, p2, p1, None, None],
                         [p2, p1, p1, None, None, None],
                         [p2, p1, None, None, None, None],
                         [None, None, None, None, None, None],
                         [p1, p1, None, None, None, None],
                         [p1, None, None, None, None, None]]
    solver = MinMaxSolver()
    # One diagonal is counted as two but it's ok
    # It is a better case than just one ordinary one
    assert solver.count_diagonal_NW_SE(game.state, p1) == 2


def test_count_diagonal_NE_SW_none_top_right():
    """
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][ ][ ][ ][a]
    [ ][ ][a][ ][ ][a][b]
    [ ][ ][a][ ][a][b][b]
    [ ][ ][b][ ][b][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    game.state.fields = [[None, None, None, None, None, None],
                         [None, None, None, None, None, None],
                         [p2, p1, p1, None, None, None],
                         [None, None, None, None, None, None],
                         [p2, p1, None, None, None, None],
                         [p1, p2, p1, None, None, None],
                         [p1, p2, p2, p1, None, None]]
    solver = MinMaxSolver()
    assert solver.count_diagonal_NE_SW(game.state, p1) == 1


def test_count_diagonal_NE_SW_none_left_bottom():
    """
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][a][ ][ ][a][b]
    [ ][ ][a][ ][a][b][b]
    [ ][ ][b][a][b][a][a]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    game.state.fields = [[None, None, None, None, None, None],
                         [None, None, None, None, None, None],
                         [p2, p1, p1, None, None, None],
                         [p1, None, None, None, None, None],
                         [p2, p1, None, None, None, None],
                         [p1, p2, p1, None, None, None],
                         [p1, p2, p2, None, None, None]]
    solver = MinMaxSolver()
    assert solver.count_diagonal_NE_SW(game.state, p1) == 1


def test_count_diagonal_NE_SW_double_case():
    """
    Characteristic case
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][ ][ ][ ][ ]
    [ ][ ][ ][ ][ ][a][ ]
    [ ][a][ ][ ][a][b][ ]
    [ ][a][ ][a][b][b][ ]
    [ ][b][a][b][a][a][ ]
    """
    p1 = Player("a")
    p2 = Player("b")
    game = ConnectFour(size=(COLUMN_COUNT, ROW_COUNT), first_player=p1, second_player=p2)
    game.state.fields = [[None, None, None, None, None, None],
                         [p2, p1, p1, None, None, None],
                         [None, None, None, None, None, None],
                         [p2, p1, None, None, None, None],
                         [p1, p2, p1, None, None, None],
                         [p1, p2, p2, p1, None, None],
                         [None, None, None, None, None, None]]
    solver = MinMaxSolver()
    # One diagonal is counted as two but it's ok
    # It is a better case than just one ordinary one
    assert solver.count_diagonal_NE_SW(game.state, p1) == 2
