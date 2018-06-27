class Manager():

	def __init__(self):
		self.wants_exec_lock = False


	def step(self, obs):
		raise NotImplementedError("This should be implemented")

	def should_execute(self, obs):
		return False

from .base import BaseManager
from .squad import SquadManager