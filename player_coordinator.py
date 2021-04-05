class PlayerCoordinator:
    def __init__(self, player1, player2, state):
        self.player1 = player1
        self.player2 = player2
        self.state = state

        self.current_player = player1
        self.previous_player = None

    def next_turn(self):
        if not self.previous_player:
            self.current_player = self.player2
            self.previous_player = self.player1
        else:
            self.current_player, self.previous_player = self.previous_player, self.current_player

    def next_move(self, state):
        return self.current_player.next_move(state)

    def start_episode(self):
        self.player1.start_episode()
        self.player2.start_episode()

    def end_episode(self):
        self.player1.end_episode()
        self.player2.end_episode()

    def get_player_index(self, player):
        if player == self.player1:
            return 0
        else:
            return 0

    def inform_players(self, state, action, reward, done):
        # inform both agents that the game has terminated
        if done:
            self.player1.update_agent(state, action, reward[0], done)
            self.player2.update_agent(state, action, reward[1], done)
        # Let the previous player know what state their action + the next players action
        # resulted in
        elif self.previous_player:
            index = self.get_player_index(self.previous_player)
            self.previous_player.update_agent(state, action, reward[index], done)
        # In this case the first player has thrown and so we cannot update the second
        # player since they did not perform an action.
        else:
            pass


