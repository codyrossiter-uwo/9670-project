def state_to_string(state):
    """
    Convert the state from a pair (round, grid) to a string representation.
    :param state: The (round, grid) pair representing the current game state.
    :return: A string representing the state.
    """
    round, grid = state
    state_string = str(round)

    for row in grid:
        for value in row:
            state_string += str(value)

    return state_string