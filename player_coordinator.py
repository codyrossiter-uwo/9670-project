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

    def next_move(self):
        return self.current_player.next_move()

    def inform_player(self, state, reward, done):
        # self.agent.process_move(...)
        pass

