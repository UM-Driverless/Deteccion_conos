from controller_agent.agent import AgentAccelerationYoloFast
from trayectory_estimation.cone_processing import ConeProcessingTest180

class AgentEBS(AgentAccelerationYoloFast):
    def __init__(self, logger, target_speed=45.):
        super().__init__(logger=logger, target_speed=45.)

class AgentTest180(AgentAccelerationYoloFast):
    def __init__(self, logger, target_speed=5.):
        super().__init__(logger=logger, target_speed=target_speed, intialTrackVar = [2, 0, 0, 30, 0, 0, 10, 25, 40])
        self.cone_processing = ConeProcessingTest180()

