import numpy as np
from pysc2.lib import features, actions, units
from scipy import spatial

from collections import defaultdict
import build_orders
from managers import Manager
from utils import Location
from utils import locate_deposits, get_mean_player_position, noops, what_builds
import random

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_HIT_POINT_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

_SELECT_POINT = actions.FUNCTIONS.select_point.id

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_COMMAND_CENTER = units.Terran.CommandCenter

BASE_EMPTY = 0
BASE_SELF = 1
BASE_ENEMY = 2

_NOT_QUEUED = [False]
_QUEUED = [True]


class Base():
	def __init__(self, location, ownership):
		self.location = location
		self.ownership = ownership
		self.production_queue = None
		self.buildings = defaultdict(list)

	def __repr__(self):
		return "{}: {}".format(['EMPTY', 'PLAYER', 'ENEMY'][self.ownership], self.location)


	def init_production(self, build_order):
		if build_order is None:
			return
		self.production_queue = build_order

	def produce(self, unit):
		print("Producing: {}".format(str(unit)))
		building, ability = what_builds(unit)
		if self.buildings[building]:
			location = self.buildings[building][0]
			return [
				*location.select(),
				actions.FunctionCall(ability, [_QUEUED])
			]
		else:
			return []

	def can_produce(self):
		pass

	def manage_production(self, obs):
		assert self.ownership == BASE_SELF

		if self.production_queue:
			item = self.production_queue[0]
			self.production_queue = []
			return self.produce(item)

		return noops(1)

class BaseManager(Manager):
	def __init__(self, obs):
		super().__init__()
		self.obs = obs
		np.set_printoptions(threshold=np.nan)

		bases = locate_deposits(obs).tolist()  # possible bases (minimap coords)

		tree = spatial.KDTree(bases)
		starting_base_index = tree.query([get_mean_player_position(obs, 'feature_minimap')])[1][0]

		self.bases = [Base(Location(loc, None), BASE_EMPTY) for loc in bases]
		self.bases[starting_base_index].ownership = BASE_SELF
		self.starting_base = self.bases[starting_base_index]
		self.starting_base.init_production(build_orders.TERRAN_MARINE_RUSH)

		# the starting base already has a command center / add it to the base's producers
		command_y, command_x = (
				obs.observation['feature_screen'][_UNIT_TYPE] == _COMMAND_CENTER
		).nonzero()
		command_y, command_x = np.median(command_y), np.median(command_x)
		self.starting_base.buildings[_COMMAND_CENTER].append(
			Location(self.starting_base.location.minimap,(command_x, command_y))
		)
		#####


	def should_execute(self, obs):
		return True

	def step(self, obs):
		self.obs = obs

		return [
			*self.starting_base.location.zoom(),
			*self.starting_base.manage_production(obs),
		 ]

