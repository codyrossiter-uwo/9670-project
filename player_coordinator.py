class PlayerCoordinator:
    def __init__(self, player1, player2, state):
        self.player1 = player1
        self.player2 = player2
        self.state = state

        self.current_player = player1

    def next_turn(self):
        if self.current_player == self.player1:
            self.current_player = self.player2
        else:
            self.current_player = self.player1

    def next_move(self, state):
        return self.current_player.next_move(state)

    def start_episode(self):
        self.player1.start_episode()
        self.player2.start_episode()

    def end_episode(self):
        self.player1.end_episode()
        self.player2.end_episode()

    def inform_player(self, state, action, reward, done):
        # self.agent.process_move(...)
        pass

