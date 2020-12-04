"""
Classic cart-pole system implemented by Rich Sutton et al.
Base copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnvContinuous(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts downright, and the goal is to
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
        Type: Continuous
        Num   Action
        -1 to 1, how much of force magnitude is applied        
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Possibly if cartpole falls from upward position below center line.
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    """
    kill_on_fall : changes done to true if pole has been upward and falls beyond
                   center line
    reward_model : 
        0: default reward 1.0
        1: Reward based on reward regions
           Give 3 times reward if pole is 30deg from up position and
           another 3 times reward if cart is within half the distance of
           track from center        
    """

    def __init__(self, kill_on_fall=True, reward_model=0, convert_theta_to_cos_sin=False, add_noise_to_state=False, randomise_reset_state=False, poke_cart=False):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle and x at which to fail the episode
        self.theta_threshold_radians = np.finfo(np.float32).max
        self.x_threshold = 2.4

        # What is considered upward position
        self.upward_position = np.pi / 3.0  # 30 deg

        # Convert theta value to it's sin and cos in state vector
        self.convert_theta_to_cos_sin = convert_theta_to_cos_sin

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, ), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.has_been_up = False

        self.kill_on_fall = kill_on_fall
        self.reward_model = reward_model

        self.add_noise_to_state = add_noise_to_state

        self.randomise_reset_state = randomise_reset_state

        # Robustness check
        self.poked = 0
        self.poke_cart = poke_cart

    def get_state_dim(self):
        state_dim = self.observation_space.shape[0]
        if (self.convert_theta_to_cos_sin):
            state_dim += 1
        return state_dim

    def get_action_dim(self):
        return self.action_space.shape[0]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state_vector(self):
        if (self.convert_theta_to_cos_sin):
            costheta = math.cos(self.state[2])
            sintheta = math.sin(self.state[2])
            state_vector = np.array(
                [self.state[0], self.state[1], costheta, sintheta, self.state[3]])
        else:
            state_vector = np.array(self.state)

        if (self.add_noise_to_state):
            noise = np.random.normal(0, 0.01, len(state_vector))
            state_vector += noise

        return state_vector

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = action[0] * self.force_mag

        # Check robustness by "poking" the cart by max force
        if (self.poke_cart):
            if (np.random.random() > 0.95):
                self.poked = 20  # self.force_mag
                force = force - self.poked
            else:
                self.poked = 0

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot **
                2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length *
                                                                  (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
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

        theta_norm = np.abs(theta) % (2.0 * np.pi)
        # Keep track of if pole has been upward or not
        if (theta_norm < self.upward_position and not self.has_been_up):
            self.has_been_up = True

        has_been_up_and_fallen = np.abs(theta) > np.pi and self.has_been_up

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or (self.kill_on_fall and has_been_up_and_fallen)
        )

        if not done:
            if (self.reward_model == 0):
                reward = 1.0
            elif (self.reward_model == 1):
                # Reward based on reward regions
                # Give 3 times reward if pole is 30deg from up position and
                # another 3 times reward if cart is within half the distance of
                # track from center
                reward = 1.0
                if (theta_norm < self.upward_position):
                    reward *= 3.0
                if (theta_norm < self.x_threshold / 2.0):
                    reward *= 3.0
            else:
                reward = 1.0

        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self.get_state_vector(), reward, done, {}

    def reset(self):
        if (self.randomise_reset_state):
            self.state = self.np_random.uniform(low=-1.0, high=1.0, size=(4,))
        else:
            self.state = self.np_random.uniform(
                low=-0.05, high=0.05, size=(4,))

        self.state[2] += np.pi
        self.steps_beyond_done = None
        self.has_been_up = False

        return self.get_state_vector()

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
            l, r, t, b = -polewidth / 2, polewidth / \
                2, polelen - polewidth / 2, -polewidth / 2
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

        if (self.poked != 0):
            from gym.envs.classic_control import rendering
            print('poked: {}N'.format(self.poked))
            dx = cartwidth / 2
            l = 45
            h = 30
            arrow = rendering.FilledPolygon(
                [(dx, 0), (dx + h/2, l/3), (dx + h/2, h/4), (dx + l, h/4), (dx + l, -h/4), (dx + l/3, -h/4), (dx + l/3, -h/2)])
            # arrow.add_attr(self.poletrans)
            arrow.add_attr(self.carttrans)
            arrow.set_color(1.0, 0.0, 0.0)
            self.viewer.add_onetime(arrow)

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / \
            2, polelen - polewidth / 2, -polewidth / 2
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
