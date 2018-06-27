class GameData(object):
	def __init__(self, static_data):
		self._data = static_data

		self.units = {u.unit_id: UnitTypeData(self, u) for u in self._data.units if u.available}
		self.abilities = {a.ability_id: AbilityData(self, a) for a in self._data.abilities if a.available}

class UnitTypeData(object):

	def __init__(self, game_data, proto):
		self._game_data = game_data
		self._proto = proto

	@property
	def creation_ability(self):
		if self._proto.ability_id == 0:
			return None
		if self._proto.ability_id not in self._game_data.abilities:
			return None
		return self._game_data.abilities[self._proto.ability_id]


class AbilityData(object):

	def __init__(self, game_data, proto):
		self._game_data = game_data
		self._proto = proto

	@property
	def id(self):
		if self._proto.remaps_to_ability_id:
			return self._proto.remaps_to_ability_id
		return self._proto.ability_id

	def __repr__(self):
		return f"AbilityData(name={self._proto.button_name})"
