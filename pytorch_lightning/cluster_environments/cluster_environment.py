class ClusterEnvironment:

    def __init__(self, world_size):
        self._world_size = world_size

    def master_address(self):
        pass

    def master_port(self):
        pass

    def world_size(self):
        return self._world_size

