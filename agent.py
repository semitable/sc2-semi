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

class ActionQueue:

	def __init__(self):
		self._queue = []

	def is_empty(self):
		return not self._queue

	def reset(self):
		self._queue = []

	def queue(self, action):
		"""
		queues an action to the action queue
		:param action: the action(s) to be queued. Can be a single action or a list
		"""
		print("Queuing: ", action)

		if isinstance(action, actions.FunctionCall):
			self._queue.append(action)
		else:
			self._queue += action


	def dequeue(self):
		action = None
		try:
			while action is None:
				action = self._queue.pop(0)
		except IndexError:
			return actions.FunctionCall(_NO_OP, [])

		return action




class SemiAgent(BaseAgent):


	def __init__(self):
		super().__init__()
		self.squad_manager = SquadManager()
		self.action_queue = ActionQueue()


	def step(self, obs):
		super().step(obs)

		if obs.first():
			self.action_queue.reset()

		if self.action_queue.is_empty():
			self.action_queue.queue(self.squad_manager.step(obs))

		action = self.action_queue.dequeue()
		if action.function in obs.observation['available_actions']:
			return action
		else:
			return actions.FunctionCall(_NO_OP, [])