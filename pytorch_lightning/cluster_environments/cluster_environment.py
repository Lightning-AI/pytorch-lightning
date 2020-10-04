class ClusterEnvironment:

    def __init__(self):
        self._world_size = None

    def master_address(self):
        pass

    def master_port(self):
        pass

    def world_size(self):
        return self._world_size
