import numpy as np
from matplotlib import pyplot as plt
from pysc2.lib import features
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

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

class Location:
	def __init__(self, minimap, screen):
		self.minimap = minimap
		self.screen = screen


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