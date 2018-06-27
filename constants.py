from pysc2.lib.units import Terran
from pysc2.lib import actions

what_builds = {
	Terran.SCV: ([Terran.CommandCenter, Terran.OrbitalCommand], actions.FUNCTIONS.Train_SCV_quick.id),
	Terran.Marine: ([Terran.Barracks], actions.FUNCTIONS.Train_SCV_quick.id),
}