from utils import Location
import numpy as np
from pysc2.lib import actions
from pysc2.lib import features
from utils import locate_deposits, get_mean_player_position
from scipy import spatial
from matplotlib import pyplot as plt

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

class BaseManager:
	def __init__(self, obs):
		self.obs = obs
		np.set_printoptions(threshold=np.nan)

		self.bases = locate_deposits(obs).tolist() # possible bases (minimap coords)

		tree = spatial.KDTree(self.bases)
		self.starting_base = self.bases[
			tree.query([get_mean_player_position(obs, 'feature_minimap')])[1][0]
		]

	def step(self, obs):
		self.obs = obs


