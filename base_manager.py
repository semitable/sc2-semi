import numpy as np
from pysc2.lib import features
from scipy import spatial

from utils import Location
from utils import locate_deposits, get_mean_player_position

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_HIT_POINT_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

BASE_EMPTY = 0
BASE_SELF = 1
BASE_ENEMY = 2


class Base():
	def __init__(self, location, ownership):
		self.location = location
		self.ownership = ownership

	def __repr__(self):
		return "{}: {}".format(['EMPTY', 'PLAYER', 'ENEMY'][self.ownership], self.location)


class BaseManager:
	def __init__(self, obs):
		self.obs = obs
		np.set_printoptions(threshold=np.nan)

		bases = locate_deposits(obs).tolist()  # possible bases (minimap coords)

		tree = spatial.KDTree(bases)
		starting_base_index = tree.query([get_mean_player_position(obs, 'feature_minimap')])[1][0]

		self.bases = [Base(Location(loc, None), BASE_EMPTY) for loc in bases]
		self.bases[starting_base_index].ownership = BASE_SELF

	def step(self, obs):
		self.obs = obs
