import pickle

import numpy as np
from pysc2 import run_configs, maps
from pysc2.lib import actions
from pysc2.lib import features
from s2clientprotocol import (
	sc2api_pb2 as sc_pb,
	common_pb2 as sc_common
)
from sklearn.cluster import MeanShift, estimate_bandwidth

STATIC_DATA_PICKLE_PATH = 'C:\\Users\\AlMak Semitable\\Documents\\GitHub\\starcraft\\sc2-semi\\static_data.pk'

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_HIT_POINT_RATIO = features.SCREEN_FEATURES.unit_hit_points_ratio.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

BASE_EMPTY = 0
BASE_SELF = 1
BASE_ENEMY = 2

_NOT_QUEUED = [False]
_QUEUED = [True]


class Location:
	def __init__(self, minimap, screen):
		self.minimap = minimap
		self.screen = screen

	def __repr__(self):
		return "{}-{}".format(self.minimap, self.screen)

	def zoom(self):
		return [
			actions.FunctionCall(_MOVE_CAMERA, [[self.minimap[1], self.minimap[0]]]),
		]

	def select(self):
		return [
			actions.FunctionCall(_MOVE_CAMERA, [[self.minimap[1], self.minimap[0]]]),
			actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, self.screen]),
		]


def get_static_data():
	"""Retrieve static data from the game."""

	try:
		with open(STATIC_DATA_PICKLE_PATH, 'rb') as f:
			static_data = pickle.load(f)
		return static_data
	except FileNotFoundError:
		pass

	run_config = run_configs.get()

	with run_config.start() as controller:
		m = maps.get("Sequencer")  # Arbitrary ladder map.
		create = sc_pb.RequestCreateGame(local_map=sc_pb.LocalMap(
			map_path=m.path, map_data=m.data(run_config)))
		create.player_setup.add(type=sc_pb.Participant)
		create.player_setup.add(type=sc_pb.Computer, race=sc_common.Random,
								difficulty=sc_pb.VeryEasy)
		join = sc_pb.RequestJoinGame(race=sc_common.Random,
									 options=sc_pb.InterfaceOptions(raw=True))

		controller.create_game(create)
		controller.join_game(join)
		static_data = controller.data_raw()

		with open(STATIC_DATA_PICKLE_PATH, 'wb') as f:
			static_data = pickle.dump(static_data, f, protocol=pickle.HIGHEST_PROTOCOL)
		return static_data


def noops(N):
	"""
	returns a list with N no ops
	"""
	return [actions.FunctionCall(_NO_OP, []) for _ in range(N)]


def locate_deposits(obs):
	X = np.argwhere(
		(obs.observation['feature_minimap'][_PLAYER_RELATIVE] == _PLAYER_NEUTRAL)
	)

	bandwidth = estimate_bandwidth(X, quantile=0.05)

	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	# labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	# labels_unique = np.unique(labels)
	# n_clusters_ = len(labels_unique)
	#
	# print("number of estimated clusters : %d" % n_clusters_)
	#
	# plt.figure(1)
	# plt.clf()
	#
	# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	# for k, col in zip(range(n_clusters_), colors):
	# 	my_members = labels == k
	# 	cluster_center = cluster_centers[k]
	# 	plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
	# 	plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
	# 			 markeredgecolor='k', markersize=14)
	# plt.title('Estimated number of clusters: %d' % n_clusters_)
	# plt.show()

	return np.rint(cluster_centers).astype(int)


def get_mean_player_position(obs, feature):
	player_y, player_x = (obs.observation[feature][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()

	if not player_y.any():
		return None

	return (
		int(player_x.mean()),
		int(player_y.mean())
	)
