import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_SELECT_NO_ADD = [False]
_SELECT_ADD = [True]


class SquadManager():
	"""
	This is the squad manager class; It controls the units currently on screen
	"""

	def __init__(self):
		self.obs = None

	def should_engage(self):
		return True

	def should_flee(self):
		return False

	def should_select(self):
		selected = self.obs.observation['feature_screen'][_SELECTED]
		player = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_SELF)

		return (player-selected).any()


	def select_onscreen(self):

		player_y, player_x = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

		min_y = player_y.min()
		max_y = player_y.max()

		min_x = player_x.min()
		max_x = player_x.max()

		return actions.FunctionCall(_SELECT_RECT, [_SELECT_NO_ADD, [min_x, min_y], [max_x, max_y]])

	def step(self, obs):

		self.obs = obs

		if self.should_select():
			return self.select_onscreen()

		return actions.FunctionCall(_NO_OP, [])
