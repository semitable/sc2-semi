"""A base agent to write custom scripted agents."""

import math
import os
import random

import numpy as np
import pandas as pd

from squad_manager import SquadManager

from pysc2.agents.base_agent import BaseAgent
from pysc2.lib import actions
from pysc2.lib import features

DATA_FILE = 'sparse_agent_data'

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

class QLearningTable:
	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions  # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def choose_action(self, observation):
		self.check_state_exist(observation)

		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.ix[observation, :]

			# some actions have the same value
			state_action = state_action.reindex(np.random.permutation(state_action.index))

			action = state_action.idxmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)

		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		self.check_state_exist(s)

		q_predict = self.q_table.ix[s, a]

		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.ix[s_, :].max()
		else:
			q_target = r  # next state is terminal
		# update
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SemiAgent(BaseAgent):


	def __init__(self):
		super().__init__()
		self.squad_manager = SquadManager()


	def step(self, obs):
		super().step(obs)

		return self.squad_manager.step(obs)