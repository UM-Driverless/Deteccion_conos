from controller_agent.agent import AgentAccelerationYoloFast


class AgentEBS(AgentAccelerationYoloFast):
    def __init__(self, logger, target_speed=45.):
        super().__init__(logger=logger, target_speed=45.)
