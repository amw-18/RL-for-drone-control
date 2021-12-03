import numpy as np


class GridWorld:

    def __init__(self):
        self.pos = 0

    def reset(self):
        self.__init__()
    
    def step(self, action):
        if action == 0:
            self._move_left()
        elif action == 1:
            self._move_right()

        if self.pos == 5:
            self.reset()
            return 1, 0
        else:
            return 0, -1

    def _move_left(self):
        
        if self.pos == 0:
            self.pos = self.pos
        elif self.pos == 1:
            self.pos += 1
        else:
            self.pos -= 1

    def _move_right(self):
        
        if self.pos == 1:
            self.pos -= 1
        else:
            self.pos += 1