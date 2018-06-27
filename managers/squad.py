import numpy as np
from pysc2.lib import actions
from pysc2.lib import features

from qlearn import QLearningTable
from managers import Manager
from utils import noops

# Functions
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_RECT = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_SCAN_MOVE = actions.FUNCTIONS.Scan_Move_screen.id
_MOVE = actions.FUNCTIONS.Move_screen.id
_ATTACK = actions.FUNCTIONS.Attack_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_HIT_POINT_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_SELECT_NO_ADD = [False]
_SELECT_ADD = [True]

_NOT_QUEUED = [False]
_QUEUED = [True]

_MAX_ENEMIES = 10

_MAX_INT = 2147483647
_SCREEN_SIZE = 84

_ENGAGE_RANGE = 15


class SquadManager(Manager):
	"""
	This is the squad manager class; It controls the units currently on screen
	"""

	def __init__(self, obs):
		super().__init__()
		self.obs = obs
		self.qlearn = QLearningTable(actions=list(range(_MAX_ENEMIES)))

	def should_execute(self, obs):
		return False

	def is_engaged(self):
		enemy_y, enemy_x = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
		enemy_y, enemy_x = enemy_y.mean(), enemy_x.mean()

		selected_y, selected_x = (self.obs.observation['feature_screen'][_SELECTED]).nonzero()
		selected_y, selected_x = selected_y.mean(), selected_x.mean()

		vx = selected_x - enemy_x
		vy = selected_y - enemy_y
		vlength = np.sqrt(vx * vx + vy * vy)

		return vlength > _ENGAGE_RANGE

	def should_engage(self):
		return True

	def should_flee(self):
		return False

	def should_select(self):
		selected = self.obs.observation['feature_screen'][_SELECTED]
		player = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_SELF)

		return (player - selected).any()

	def disengage(self):

		escape_distance = 10

		enemy_y, enemy_x = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
		enemy_y, enemy_x = enemy_y.mean(), enemy_x.mean()

		selected_y, selected_x = (self.obs.observation['feature_screen'][_SELECTED]).nonzero()
		selected_y, selected_x = selected_y.mean(), selected_x.mean()

		vx = selected_x - enemy_x
		vy = selected_y - enemy_y
		vlength = np.sqrt(vx * vx + vy * vy)

		escape_x = np.clip(int(vx / vlength * escape_distance + selected_x), 0, _SCREEN_SIZE - 1)
		escape_y = np.clip(int(vy / vlength * escape_distance + selected_y), 0, _SCREEN_SIZE - 1)

		return [
			actions.FunctionCall(_MOVE, [_NOT_QUEUED, [escape_x, escape_y]])
		]

	def select_onscreen(self):

		player_y, player_x = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

		if not player_y.any():
			return []

		min_y = player_y.min()
		max_y = player_y.max()

		min_x = player_x.min()
		max_x = player_x.max()

		return [
			actions.FunctionCall(_SELECT_RECT, [_SELECT_NO_ADD, [min_x, min_y], [max_x, max_y]])
		]

	def engage(self):
		enemy_y, enemy_x = (self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
		return [
			actions.FunctionCall(_ATTACK, [_NOT_QUEUED, [np.median(enemy_x), np.median(enemy_y)]])
		]

	def disengage_damaged(self):

		dmg_threshold = 28

		# print(self.obs.observation['last_actions'])
		# self.obs.observation['multi_select']
		ms = np.array(self.obs.observation['multi_select'])

		if ms.size == 0:
			return []

		if ms[:, 2].min() >= dmg_threshold:
			return []  # no one is damaged

		return [
			actions.FunctionCall(_SELECT_UNIT, [[0], [ms[:, 2].argmin()]]),
			*self.disengage(),
			*noops(7)
		]

	def target_lowest_hp(self):
		hostiles = self.obs.observation['feature_screen'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE

		health = self.obs.observation['feature_screen'][_HIT_POINT_RATIO]

		hostile_hp = hostiles * health

		hostile_hp = np.array(hostile_hp)
		hostile_hp[hostile_hp == 0] = _MAX_INT

		enemy_y, enemy_x = np.unravel_index(np.argmin(hostile_hp, axis=None), hostile_hp.shape)

		return [
			actions.FunctionCall(_ATTACK, [_NOT_QUEUED, [enemy_x, enemy_y]])
		]

	def micro(self):
		action_list = []
		# action_list += self.disengage_damaged()
		# action_list += self.select_onscreen()
		action_list += self.target_lowest_hp()

		return action_list

	def step(self, obs):

		self.obs = obs

		if self.should_select():
			return self.select_onscreen()

		is_engaged = self.is_engaged()

		if is_engaged and self.should_flee():
			return self.disengage()

		if not is_engaged and self.should_engage():
			return [
				*self.select_onscreen(),
				*self.engage()
			]

		# todo: rest of engagement cases
		# Battle micro:
		return self.micro()
