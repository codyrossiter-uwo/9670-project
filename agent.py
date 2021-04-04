class Agent:
    """
    This is the base class for agents using the curling environment.
    The PlayerCoordinator class will use these methods to interact with
    the environment.
    """
    def __init__(self, name):
        self.name = name
        # TODO: add support for reading/saving policy data from file
        # TODO: add support for learning mode/playing mode

    def next_move(self):
        """
        Get the next action from the agent based on the current state.
        :return: The agent's next action.
        """
        raise NotImplementedError()

    def update_agent(self, state, reward, done):
        """
        Inform the agent the change in the environment based on the last action.
        :param state: The new state.
        :param reward: The reward from the previous action.
        :param done: A boolean representing if the episode is complete.
        :return: None
        """
        raise NotImplementedError()
