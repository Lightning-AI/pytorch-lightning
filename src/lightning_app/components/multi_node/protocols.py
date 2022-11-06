from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class DistributedProtocol(Protocol):
    def run(
        self,
        main_address: str,
        main_port: int,
        num_nodes: int,
        node_rank: int,
    ) -> None:
        pass


@runtime_checkable
class LiteProtocol(Protocol):
    @staticmethod
    def run(lite) -> None:
        ...


@runtime_checkable
class DistributedPyTorchSpawnProtocol(Protocol):
    def run(
        self,
        world_size: int,
        node_rank: int,
        global_rank: int,
        local_rank: int,
    ) -> None:
        pass
