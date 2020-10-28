from gym.envs.classic_control import rendering

class Obstacle:
    """
    Description:
        An obstacles that causes heavy penalties when hit by the
        cartpole pole
    """
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.color = [1.0, 0.0, 0.0]

    def set_color(self, r, g, b):
        self.color = [r, g, b]

    def get_geom(self):
        obstacle = rendering.FilledPolygon([(self.left, self.bottom), (self.left, self.top), 
        (self.right, self.top), (self.right, self.bottom)])
        obstacle.set_color(self.color[0], self.color[1], self.color[2])
        return obstacle

    def hit(self, point):
        x, y = point
        if (x < self.right and x > self.left):
            if (y < self.top and y > self.bottom):
                return True
        return False
