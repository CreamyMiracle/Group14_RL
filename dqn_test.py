from datetime import datetime
start_time = datetime.now() # Start timing the program
import sys
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from cartpole_env_2 import CartPoleModEnv # Import modded version of cartpole

TIME_STEPS = 60000
TRIALS = 1000
RUNS = 100
# np.random.seed(10) # Set appropriate seed value


class deepQNetwork:
	def __init__(self, state_size, action_size):

		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95	# discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.9
		self.learning_rate = 0.01
		self.model = self._build_model()
		self.render = False

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
		model.add(Dense(48, activation='tanh'))
		model.add(Dense(self.action_size, activation='linear'))
		print('Compiling Neural Network...')
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def set_epsilon(self,epsilon):
		# Set exploration rate
		self.epsilon = epsilon

	def reset_memory(self):
		# Reset memory and rebuild the neural network
		self.memory = deque(maxlen=2000)
		self.set_epsilon(1.0)
		self.model = self._build_model()

	def remember(self, state, action, reward, next_state, done):
		# Save the states for the Q function(target)
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		# Take random action if epsilon is high, else, take action based on the predicted states
		if self.epsilon > 0.01:
			if np.random.rand() <= self.epsilon:
				return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	def replay(self, batch_size):
		# Calculate the Q function (Target) for the saved states
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0) # Train the model for each state
		# Update the exploration rate for every trial/episode
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def main(self,explore=True):
		# Do not explore if explore=False
		if not explore:
			self.set_epsilon(0.01)
		# Initialize parameters
		done = False
		batch_size = 64
		trial_score = deque(maxlen = RUNS)
		run_success = deque(maxlen = RUNS)
		total_success = 0
		for run in range(1,RUNS+1):
			# Reset memory for every run
			self.reset_memory()
			scores = deque(maxlen = 100)
			success = 0
			for trial in range(1,TRIALS+1):
				# Reset states for each trial
				state = env.reset()
				state = np.reshape(state, [1, state_size])
				total_reward = 0
				for time in range(TIME_STEPS):
					# Render environment for visualization. Uncomment if needed.
					if (self.render):
						env.render()
					action = self.act(state)
					next_state, reward, done, _ = env.step(action, time)
					reward = reward if not done else -10 # Add a negative reward if failed
					next_state = np.reshape(next_state, [1, state_size])
					self.remember(state, action, reward, next_state, done) # Save states for each time step
					state = next_state
					if done:
						scores.append(time) # Calculate a score based on the time-steps reached
						if trial%100 ==0:
							print('Run:{} Trial:{}, Mean score over 100 trials: {}'.format(run,trial,np.mean(scores)))
						break
					elif time== TIME_STEPS/10:
						print('It\'s gonna be a great trial! Ran {} times already! Hope Trial#{} goes on!'.format(TIME_STEPS/10,trial))
					elif time>=(TIME_STEPS-1):
						scores.append(time)
						print('Woah!!!!')
						print('Run:{} Trial:{}, Time: {}'.format(run,trial,time+1))
						success = 1
						break
				try:
					mean_score = np.mean(scores)
				except:
					mean_score = 0
				if success==1:
					trial_score.append(trial)
					print('Successful trial. Run#:{}'.format(run))
					break
				if len(self.memory) > batch_size:
					self.replay(batch_size)
			run_success.append(success)
			try:
				mean_trial = np.mean(trial_score)
			except:
				mean_trial = 0
			total_success += success
			if success==0:
				print('\t\t\t\tFailed Run#:{}'.format(run))
			else:
				print('Successful run#: {} Average trial#: {} Successes till now:{}'.format(run,mean_trial,total_success))
		print('\n\n\n\n\n\nSuccess Rate:{}% #Trials: {}'.format(total_success/RUNS*100,mean_trial))


if __name__ == "__main__":
	print('Making CartpoleMod environment')
	env = CartPoleModEnv()
	env._max_episode_steps = TIME_STEPS
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = deepQNetwork(state_size, action_size)
	agent.main()
	print('---------Execution time: {} for Case {}------------'.format(datetime.now()-start_time,sys.argv[1]))