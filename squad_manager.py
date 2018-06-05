import numpy as np
from matplotlib import pyplot as plt
from pysc2.lib import actions
from pysc2.lib import features
from qlearn import QLearningTable

# Functions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_SCAN_MOVE = actions.FUNCTIONS.Scan_Move_screen.id
_ATTACK = actions.FUNCTIONS.Attack_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_HIT_POINT_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_SELECT_NO_ADD = [False]
_SELECT_ADD = [True]

_NOT_QUEUED = [False]
_QUEUED = [True]

_MAX_ENEMIES = 10

_MAX_INT = 2147483647

class SquadManager():
	"""
	This is the squad manager class; It controls the units currently on screen
	"""

	def __init__(self):
		self.obs = None
		self.is_engaged = False

		self.qlearn = QLearningTable(actions=list(range(_MAX_ENEMIES)))

	@property
	def is_engaged(self):
		return self._is_engaged

	@is_engaged.setter
	def is_engaged(self, value):
		self._is_engaged = value

	def should_engage(self):
		return True

	def should_flee(self):
		return False

	def should_select(self):
		selected = self.obs.observation['feature_screen'][_SELECTED]
		player = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_SELF)

		return (player - selected).any()

	def select_onscreen(self):

		player_y, player_x = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

		min_y = player_y.min()
		max_y = player_y.max()

		min_x = player_x.min()
		max_x = player_x.max()

		return actions.FunctionCall(_SELECT_RECT, [_SELECT_NO_ADD, [min_x, min_y], [max_x, max_y]])

	def engage(self):
		enemy_y, enemy_x = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
		self.is_engaged = True

		return actions.FunctionCall(_ATTACK, [_NOT_QUEUED, [np.median(enemy_x), np.median(enemy_y)]])

	def micro(self):
		hostiles = self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE

		health = self.obs.observation['feature_screen'][_HIT_POINT_RATIO]

		hostile_hp = hostiles * health

		hostile_hp = np.array(hostile_hp)
		hostile_hp[hostile_hp == 0] = _MAX_INT

		enemy_y, enemy_x = np.unravel_index(np.argmin(hostile_hp, axis=None), hostile_hp.shape)
		return actions.FunctionCall(_ATTACK, [_NOT_QUEUED, [enemy_x, enemy_y]])

	def step(self, obs):

		self.obs = obs

		if self.should_select():
			return self.select_onscreen()

		if self.is_engaged and self.should_flee():
			raise NotImplementedError("Fleeing from battle is not yet implemented")

		if not self.is_engaged and self.should_engage():
			return self.engage()

		# todo: rest of engagement cases

		# Battle micro:
		return self.micro()
