import gym
from mpc import MPC
import numpy as np
import sys
import warnings as warn
from cartpole_env import CartPoleEnv
import time

env = CartPoleEnv()
try:
    env.theta_threshold_radians = np.pi * 2
    #env.x_threshold = 10
    env.length = 0.5
    start_theta = 0#-np.pi + 0.4# + 0.1#2 * np.pi #np.pi+0.4
    
    mpc = MPC(0.5,0,start_theta,0) 
    action = 0
    time.sleep(5)
    for i_episode in range(1):
        
        observation = env.reset()
        env.state[2] = start_theta - np.pi
        env.render()
        for t in range(600):
            env.render()
            time.sleep(0.05)
            #print(observation)
            #action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            a = mpc.update(observation[0], observation[1], observation[2]+np.pi, observation[3])
            env.force_mag = abs(a) #min(100, abs(a))
            #print(env.potential_energy)
            #print(env.kinematic_energy)
            #print(env.rotational_energy)
            print(reward)
            if a < 0:
            	action = 0
            else:
            	action = 1
            if done:
            	pass
        env.close()
except:
    env.close()
    print("failed")