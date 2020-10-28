"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

theta_reward_treshold_deg = 12
theta_treshold_deg = 360 # 360 is the whole circle

class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        1st model: Reward is 1 for every step taken, including the termination step
        2nd model: Reward is determined by angle of the pole further from straight equals bad
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }    

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = theta_treshold_deg * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        
        self.reward_range = (-float('inf'), float('inf'))
        self.max_steps = 10000

        # Pole starting position 0 = up, 1 = down
        self.starting_position = 1

        # Reward model 0 = one or zero, 1 = pole angle based, 2 / 3 = test
        self.reward_model = 7

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward(self, done):
        reward = 0.0
        if not done or self.steps_beyond_done is None:
            if self.steps_beyond_done is None:                
                self.steps_beyond_done = 0

            # Default
            if self.reward_model == 0:
                reward = 1.0

            # Pole angle based
            elif self.reward_model == 1: 
                theta = self.state[2]
                x = self.state[0]
                theta_reward_treshold_rad =  theta_reward_treshold_deg * 2 * math.pi / 360
                #reward = (theta_reward_treshold_rad - theta) * 100 #- np.power(10, np.abs(x))
                theta_diff = np.abs(np.pi - theta)
                reward = 0
                if theta_diff < theta_reward_treshold_rad:
                    reward += 10.0 / theta_diff
                reward /= np.abs(x)
            
            # Testing: works when starting from top
            elif self.reward_model == 2: 
                theta_diff = np.abs(self.state[2])
                reward = np.abs(np.cos(theta_diff) * 2.0 * self.length)
                if (theta_diff > np.pi / 2.0):
                    reward = -reward

            # Quite good intuitive solutions but sometimes get stuck wiggling down low
            elif self.reward_model == 3: 
                theta_diff = np.abs(self.state[2])
                reward = np.abs(np.cos(theta_diff) * 2.0 * self.length)
                if (theta_diff > np.pi / 2.0):                    
                    reward = 0.0

            
            # Harder penalty test
            elif self.reward_model == 4: 
                theta_diff = np.abs(self.state[2])
                reward = np.abs(np.cos(theta_diff) * 2.0 * self.length)

                # Being down is bad
                if (theta_diff > np.pi / 2.0):                    
                    reward = -0.05
                # Being too far from center is worse    
                elif (np.abs(self.state[0]) > 0.8 * self.x_threshold):                    
                    reward = -0.1
                # Being upward is really good             
                elif (theta_diff < self.theta_threshold_radians):                    
                    reward = 1.0            


            # Based on number 3
            elif self.reward_model == 5: 
                theta_diff = np.abs(self.state[2])
                reward = np.abs(np.cos(theta_diff) * 2.0 * self.length)
                if (theta_diff > np.pi / 2.0):                    
                    reward = -0.1

                    
            # Based on number 3
            elif self.reward_model == 6: 
                theta_diff = np.abs(self.state[2])
                reward = np.abs(np.cos(theta_diff) * 2.0 * self.length)
                if (theta_diff > np.pi / 2.0):                    
                    reward = -reward / self.state[3] # High angular velocity is good
                else:
                    reward = reward / self.state[3] # Low angular velocity is good
                    
            # Test
            elif self.reward_model == 7: 
                theta_diff = np.abs(self.state[2])
                reward = 1.0 / np.abs(self.state[3])
                if (theta_diff > np.pi / 2.0):                    
                    reward = -0.01

        else:
            if self.steps_beyond_done == 0:
                """
                logger.warn(
                "You are calling 'step()' even though this "
                "environment has already returned done = True. You "
                "should always call 'reset()' once you receive 'done = "
                "True' -- any further steps are undefined behavior.")
                """
            self.steps_beyond_done += 1
            reward = 0.0

        return reward

    def step(self, action, n = 0):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            #or n > self.max_steps
            #or theta < -self.theta_threshold_radians
            #or theta > self.theta_threshold_radians
        )

        reward = self.reward(done)
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        if (self.starting_position == 1): # down
            self.state[2] = self.state[2] + np.pi
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

            """
            # Add obstacles this way
            obstacle = Obstacle(20, 80, 80, 20)
            self.viewer.add_geom(obstacle.get_geom())
            """

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None        