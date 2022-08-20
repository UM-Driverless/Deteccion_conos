from controller_agent.agent import AgentAccelerationYoloFast
from trayectory_estimation.cone_processing import ConeProcessingTest180

class AgentEBS(AgentAccelerationYoloFast):
    def __init__(self, logger, target_speed=45.):
        super().__init__(logger=logger, target_speed=target_speed)
        self.mov = True

class AgentTest180(AgentAccelerationYoloFast):
    def __init__(self, logger, target_speed=5.):
        super().__init__(logger=logger, target_speed=target_speed, intialTrackVar = [3, 0, 0, 10, 0, 0, 10, 25, 40])
        self.cone_processing = ConeProcessingTest180()
        self.mov = True

class AgentInspection(AgentAccelerationYoloFast):
    def __init__(self, logger, target_speed=0):
        super().__init__(logger=logger, target_speed=0)
        self.cone_processing = ConeProcessingTest180()
        self.mov = False