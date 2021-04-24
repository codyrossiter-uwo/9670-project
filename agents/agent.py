class Agent:
    """
    This is the base class for agents using the curling environment.
    The PlayerCoordinator class will use these methods to interact with
    the environment.
    """
    def __init__(self, name, training_mode):
        self.name = name
        self.training_mode = training_mode

    def load_data(self, filepath):
        """
        Load the agent's data (if it exists) from the agent json file.
        :param filepath: The location of the agent data file.
        :return: None
        """
        raise NotImplementedError()

    def save_data(self, filepath):
        """
        Save the agent's data to the agent json file.
        :param filepath: The location of the agent data file.
        :return: None
        """
        raise NotImplementedError()

    def start_episode(self):
        """
        Perform any work needed at the start of an episode e.g, initialize
        state and reward vectors.
        :return: None
        """
        raise NotImplementedError()

    def end_episode(self):
        """
        Perform any work needed after episodes e.g., decaying epsilon.
        :return: None
        """
        raise NotImplementedError()

    def next_move(self, state):
        """
        Get the next action from the agent based on the current state.
        :param state: The state used to determine the next action.
        :return: The agent's next action.
        """
        raise NotImplementedError()

    def update_agent(self, state, action, reward, done):
        """
        Inform the agent the change in the environment based on the last action.
        :param state: The new state.
        :param action: The action taken that caused the update
        :param reward: The reward from the previous action.
        :param done: A boolean representing if the episode is complete.
        :return: None
        """
        raise NotImplementedError()
