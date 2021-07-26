import numpy as np

MOVER_FIRST = 0
MOVER_SECOND = 0


# Soccer Environment
class Soccer:
    def __init__(self):
        # initialization according to the figure 4 of the paper

        # pos as list of 2 elements, 1st element as position of player A
        # 2nd element as position of player B
        self.pos = [np.array([0, 2]), np.array([0, 1])]
        self.ball = 1
        self.goal = [0, 3]
        # top-left corner as (0,0) origin point
        # 0:North, 1:East, 2:South, 3:West, 4:Stick
        # player can move at most one tile at a time
        # the index of actions map to movement at specific direction
        self.legal_actions = [[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]]

    def move(self, actions):

        # players action executed in random order
        # mover_first as the index 0 or 1
        # index 0 is player A, index 1 is player B
        mover_first = np.random.choice([0, 1], 1)[0]
        mover_second = 1 - mover_first

        # copy of current pos
        new_pos = self.pos.copy()

        # scores shows the reward for player A and player B
        scores = np.array([0, 0])

        # init termination status of the game
        done = 0

        # check whether the received action is valid
        if actions[0] not in range(0, 5) or actions[1] not in range(0, 5):
            # this shouldn't happen but, meh still better to check in-case agent is picking something crazy.
            print('Invalid Action, actions should be one of [0,1,2,3,4]')
            return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], scores, done
        else:
            # moving the first player
            new_pos[mover_first] = self.pos[mover_first] + self.legal_actions[actions[mover_first]]

            # check collision, 1st mover collides with 2nd mover after moving
            if (new_pos[mover_first] == self.pos[mover_second]).all():
                # if 1st mover possess ball, the ball is lost to 2nd mover
                if self.ball == mover_first:
                    self.ball = mover_second

            # no collision, update 1st mover's pos
            elif new_pos[mover_first][0] in range(0, 2) and new_pos[mover_first][1] in range(0, 4):
                self.pos[mover_first] = new_pos[mover_first]

                # if scored for player himself
                if self.pos[mover_first][1] == self.goal[mover_first] and self.ball == mover_first:  # Player scored
                    scores = ([1, -1][mover_first]) * np.array([100, -100])
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1],
                            self.ball], scores, done

                # if scored for the opponent
                elif self.pos[mover_first][1] == self.goal[mover_second] and self.ball == mover_first:
                    scores = ([1, -1][mover_first]) * np.array([-100, 100])
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1],
                            self.ball], scores, done

            # moving the second player
            new_pos[mover_second] = self.pos[mover_second] + self.legal_actions[actions[mover_second]]

            # check collision, 2nd mover collides with 1st mover after moving
            if (new_pos[mover_second] == self.pos[mover_first]).all():  # Collide
                # if 2nd mover possess ball, the ball is lost to 1st mover
                if self.ball == mover_second:
                    self.ball = mover_first

            # no collision, update 2nd mover's pos
            elif new_pos[mover_second][0] in range(0, 2) and new_pos[mover_second][1] in range(0, 4):
                self.pos[mover_second] = new_pos[mover_second]

                # if scored for player himself
                if self.pos[mover_second][1] == self.goal[mover_second] and self.ball == mover_second:
                    scores = ([1, -1][mover_second]) * np.array([100, -100])
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1],
                            self.ball], scores, done

                # if scored for the opponent
                elif self.pos[mover_second][1] == self.goal[mover_first] and self.ball == mover_second:
                    scores = np.array([-100, 100]) * [1, -1][mover_second]
                    done = 1
                    return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1],
                            self.ball], scores, done

        return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball], scores, done
