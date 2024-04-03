import inspect
import logging
import os
import signal
import sys
import time
import traceback
from functools import partial
from multiprocessing import Process
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER = bool(int(os.getenv("ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER", "0")))

if True:  # ToDo: Avoid Module level import not at top of file
    from lightning.app.core import constants
    from lightning.app.core.api import start_server
    from lightning.app.core.flow import LightningFlow
    from lightning.app.core.queues import MultiProcessQueue, QueuingSystem
    from lightning.app.storage.orchestrator import StorageOrchestrator
    from lightning.app.utilities.app_commands import run_app_commands
    from lightning.app.utilities.cloud import _sigterm_flow_handler
    from lightning.app.utilities.component import _set_flow_context, _set_frontend_context
    from lightning.app.utilities.enum import AppStage
    from lightning.app.utilities.exceptions import ExitAppException
    from lightning.app.utilities.load_app import extract_metadata_from_app, load_app_from_file
    from lightning.app.utilities.proxies import WorkRunner
    from lightning.app.utilities.redis import check_if_redis_running

if ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER:
    from lightning.app.launcher.lightning_hybrid_backend import CloudHybridBackend as CloudBackend
else:
    from lightning.app.launcher.lightning_backend import CloudBackend

if True:  # Avoid Module level import not at top of file
    from lightning.app.utilities.app_helpers import convert_print_to_logger_info
    from lightning.app.utilities.packaging.lightning_utils import enable_debugging

if hasattr(constants, "get_cloud_queue_type"):
    CLOUD_QUEUE_TYPE = constants.get_cloud_queue_type() or "redis"
else:
    CLOUD_QUEUE_TYPE = "redis"

logger = logging.getLogger(__name__)


class FlowRestAPIQueues(TypedDict):
    api_publish_state_queue: MultiProcessQueue
    api_response_queue: MultiProcessQueue


@convert_print_to_logger_info
@enable_debugging
def start_application_server(
    entrypoint_file: str, host: str, port: int, queue_id: str, queues: Optional[FlowRestAPIQueues] = None
):
    logger.debug(f"Run Lightning Work {entrypoint_file} {host} {port} {queue_id}")
    queue_system = QueuingSystem(CLOUD_QUEUE_TYPE)

    wait_for_queues(queue_system)

    kwargs = {
        "api_delta_queue": queue_system.get_api_delta_queue(queue_id=queue_id),
    }

    # Note: Override the queues if provided
    if isinstance(queues, Dict):
        kwargs.update(queues)
    else:
        kwargs.update({
            "api_publish_state_queue": queue_system.get_api_state_publish_queue(queue_id=queue_id),
            "api_response_queue": queue_system.get_api_response_queue(queue_id=queue_id),
        })

    app = load_app_from_file(entrypoint_file)

    from lightning.app.api.http_methods import _add_tags_to_api, _validate_api
    from lightning.app.utilities.app_helpers import is_overridden
    from lightning.app.utilities.commands.base import _commands_to_api, _prepare_commands

    apis = []
    if is_overridden("configure_api", app.root):
        apis = app.root.configure_api()
        _validate_api(apis)
        _add_tags_to_api(apis, ["app_api"])

    if is_overridden("configure_commands", app.root):
        commands = _prepare_commands(app)
        apis += _commands_to_api(commands)

    start_server(
        host=host,
        port=port,
        apis=apis,
        **kwargs,
        spec=extract_metadata_from_app(app),
    )


@convert_print_to_logger_info
@enable_debugging
def run_lightning_work(
    file: str,
    work_name: str,
    queue_id: str,
):
    """This staticmethod runs the specified work in the current process.

    It is organized under cloud runtime to indicate that it will be used by the cloud runner but otherwise, no cloud
    specific logic is being implemented here

    """
    logger.debug(f"Run Lightning Work {file} {work_name} {queue_id}")

    queues = QueuingSystem(CLOUD_QUEUE_TYPE)
    wait_for_queues(queues)

    caller_queue = queues.get_caller_queue(work_name=work_name, queue_id=queue_id)
    readiness_queue = queues.get_readiness_queue(queue_id=queue_id)
    delta_queue = queues.get_delta_queue(queue_id=queue_id)
    error_queue = queues.get_error_queue(queue_id=queue_id)

    request_queues = queues.get_orchestrator_request_queue(work_name=work_name, queue_id=queue_id)
    response_queues = queues.get_orchestrator_response_queue(work_name=work_name, queue_id=queue_id)
    copy_request_queues = queues.get_orchestrator_copy_request_queue(work_name=work_name, queue_id=queue_id)
    copy_response_queues = queues.get_orchestrator_copy_response_queue(work_name=work_name, queue_id=queue_id)

    run_app_commands(file)

    load_app_from_file(file)

    queue = queues.get_work_queue(work_name=work_name, queue_id=queue_id)
    work = queue.get()

    extras = {}

    if hasattr(work, "_run_executor_cls"):
        extras["run_executor_cls"] = work._run_executor_cls

    WorkRunner(
        work=work,
        work_name=work_name,
        caller_queue=caller_queue,
        delta_queue=delta_queue,
        readiness_queue=readiness_queue,
        error_queue=error_queue,
        request_queue=request_queues,
        response_queue=response_queues,
        copy_request_queue=copy_request_queues,
        copy_response_queue=copy_response_queues,
        **extras,
    )()


@convert_print_to_logger_info
@enable_debugging
def run_lightning_flow(entrypoint_file: str, queue_id: str, base_url: str, queues: Optional[FlowRestAPIQueues] = None):
    _set_flow_context()

    logger.debug(f"Run Lightning Flow {entrypoint_file} {queue_id} {base_url}")

    app = load_app_from_file(entrypoint_file)
    app.backend = CloudBackend(entrypoint_file, queue_id=queue_id)

    queue_system = app.backend.queues
    app.backend.update_lightning_app_frontend(app)
    wait_for_queues(queue_system)

    app.backend.resolve_url(app, base_url)
    if app.root_path != "":
        app._update_index_file()
    app.backend._prepare_queues(app)

    # Note: Override the queues if provided
    if queues:
        app.api_publish_state_queue = queues["api_publish_state_queue"]
        app.api_response_queue = queues["api_response_queue"]

    LightningFlow._attach_backend(app.root, app.backend)

    app.should_publish_changes_to_api = True

    storage_orchestrator = StorageOrchestrator(
        app,
        app.request_queues,
        app.response_queues,
        app.copy_request_queues,
        app.copy_response_queues,
    )
    storage_orchestrator.setDaemon(True)
    storage_orchestrator.start()

    # refresh the layout with the populated urls.
    app._update_layout()

    # register a signal handler to clean all works.
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, partial(_sigterm_flow_handler, app=app))

    if "apis" in inspect.signature(start_server).parameters:
        from lightning.app.utilities.commands.base import _prepare_commands

        _prepare_commands(app)

    # Once the bootstrapping is done, running the rank 0
    # app with all the components inactive
    try:
        app._run()
    except ExitAppException:
        pass
    except Exception:
        app.stage = AppStage.FAILED
        print(traceback.format_exc())

    storage_orchestrator.join(0)
    app.backend.stop_all_works(app.works)

    exit_code = 1 if app.stage == AppStage.FAILED else 0
    print(f"Finishing the App with exit_code: {str(exit_code)}...")

    if not exit_code:
        app.backend.stop_app(app)

    sys.exit(exit_code)


@convert_print_to_logger_info
@enable_debugging
def serve_frontend(file: str, flow_name: str, host: str, port: int):
    """This staticmethod runs the specified frontend for a given flow in a new process.

    It is organized under cloud runtime to indicate that it will be used by the cloud runner but otherwise, no cloud
    specific logic is being implemented here.

    """
    _set_frontend_context()
    logger.debug(f"Run Serve Frontend {file} {flow_name} {host} {port}")
    app = load_app_from_file(file)
    if flow_name not in app.frontends:
        raise ValueError(f"Could not find frontend for flow with name {flow_name}.")
    frontend = app.frontends[flow_name]
    assert frontend.flow.name == flow_name

    frontend.start_server(host, port)


def start_server_in_process(target: Callable, args: Tuple = (), kwargs: Dict = {}) -> Process:
    p = Process(target=target, args=args, kwargs=kwargs)
    p.start()
    return p


def format_row(elements, col_widths, padding=1):
    elements = [el.ljust(w - padding * 2) for el, w in zip(elements, col_widths)]
    pad = " " * padding
    elements = [f"{pad}{el}{pad}" for el in elements]
    return f'|{"|".join(elements)}|'


def tabulate(data, headers):
    data = [[str(el) for el in row] for row in data]
    col_widths = [len(el) for el in headers]
    for row in data:
        col_widths = [max(len(el), curr) for el, curr in zip(row, col_widths)]
    col_widths = [w + 2 for w in col_widths]
    seps = ["-" * w for w in col_widths]
    lines = [format_row(headers, col_widths), format_row(seps, col_widths, padding=0)] + [
        format_row(row, col_widths) for row in data
    ]
    return "\n".join(lines)


def manage_server_processes(processes: List[Tuple[str, Process]]) -> None:
    if not processes:
        return

    sigterm_called = [False]

    def _sigterm_handler(*_):
        sigterm_called[0] = True

    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, _sigterm_handler)

    # Since frontends run user code, any of them could fail. In that case,
    # we want to fail all of them, as well as the application server, and
    # exit the command with an error status code.

    exitcode = 0

    while True:
        # We loop until
        # 1. Get a sigterm
        # 2. All the children died but all with exit code 0
        # 3. At-least one of the child died with non-zero exit code

        # sleeping quickly at the starting of every loop
        # moving this to the end of the loop might result in some flaky tests
        time.sleep(1)

        if sigterm_called[0]:
            print("Got SIGTERM. Exiting execution!!!")
            break
        if all(not p.is_alive() and p.exitcode == 0 for _, p in processes):
            print("All the components are inactive with exitcode 0. Exiting execution!!!")
            break
        if any((not p.is_alive() and p.exitcode != 0) for _, p in processes):
            print("Found dead components with non-zero exit codes, exiting execution!!! Components: ")
            print(
                tabulate(
                    [(name, p.exitcode) for name, p in processes if not p.is_alive() and p.exitcode != 0],
                    headers=["Name", "Exit Code"],
                )
            )
            exitcode = 1
            break

    # sleeping for the last set of logs to reach stdout
    time.sleep(2)

    # Cleanup
    for _, p in processes:
        if p.is_alive():
            os.kill(p.pid, signal.SIGTERM)

    # Give processes time to terminate
    for _, p in processes:
        p.join(5)

    # clean the remaining ones.
    if any(p.is_alive() for _, p in processes):
        for _, p in processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGKILL)

        # this sleep is just a precaution - signals might take a while to get raised.
        time.sleep(1)
        sys.exit(1)

    sys.exit(exitcode)


def _get_frontends_from_app(entrypoint_file):
    """This function is used to get the frontends from the app. It will be used to start the frontends in a separate
    process if the backend cannot provide flow_names_and_ports. This is useful if the app cannot be loaded locally to
    set the frontend before dispatching to the cloud. The backend exposes by default 10 ports from 8081 if the
    app.spec.frontends is not set.

    NOTE: frontend_name are sorted to ensure that they get consistent ports.

    :param entrypoint_file: The entrypoint file for the app
    :return: A list of tuples of the form (frontend_name, port_number)

    """
    app = load_app_from_file(entrypoint_file)

    frontends = []
    # This value of the port should be synced with the port value in the backend.
    # If you change this value, you should also change the value in the backend.
    flow_frontends_starting_port = 8081
    for frontend in sorted(app.frontends.keys()):
        frontends.append((frontend, flow_frontends_starting_port))
        flow_frontends_starting_port += 1

    return frontends


@convert_print_to_logger_info
@enable_debugging
def start_flow_and_servers(
    entrypoint_file: str,
    base_url: str,
    queue_id: str,
    host: str,
    port: int,
    flow_names_and_ports: Tuple[Tuple[str, int]],
):
    processes: List[Tuple[str, Process]] = []

    # Queues between Flow and its Rest API are using multiprocessing to:
    # - reduce redis load
    # - increase UI responsiveness and RPS
    queue_system = QueuingSystem.MULTIPROCESS
    queues = {
        "api_publish_state_queue": queue_system.get_api_state_publish_queue(queue_id=queue_id),
        "api_response_queue": queue_system.get_api_response_queue(queue_id=queue_id),
    }

    # In order to avoid running this function 3 seperate times while executing the
    # `run_lightning_flow`, `start_application_server`, & `serve_frontend` functions
    # in a subprocess we extract this to the top level. If we intend to make changes
    # to be able to start these components in seperate containers, the implementation
    # will have to move a call to this function within the initialization process.
    run_app_commands(entrypoint_file)

    flow_process = start_server_in_process(
        run_lightning_flow,
        args=(
            entrypoint_file,
            queue_id,
            base_url,
        ),
        kwargs={"queues": queues},
    )
    processes.append(("Flow", flow_process))

    server_process = start_server_in_process(
        target=start_application_server,
        args=(
            entrypoint_file,
            host,
            port,
            queue_id,
        ),
        kwargs={"queues": queues},
    )
    processes.append(("Server", server_process))

    if not flow_names_and_ports:
        flow_names_and_ports = _get_frontends_from_app(entrypoint_file)

    for name, fe_port in flow_names_and_ports:
        frontend_process = start_server_in_process(target=serve_frontend, args=(entrypoint_file, name, host, fe_port))
        processes.append((name, frontend_process))

    manage_server_processes(processes)


def wait_for_queues(queue_system: QueuingSystem) -> None:
    queue_check_start_time = int(time.time())

    if hasattr(queue_system, "get_queue"):
        while not queue_system.get_queue("healthz").is_running:
            if (int(time.time()) - queue_check_start_time) % 10 == 0:
                logger.warning("Waiting for http queues to start...")
            time.sleep(1)
    else:
        while not check_if_redis_running():
            if (int(time.time()) - queue_check_start_time) % 10 == 0:
                logger.warning("Waiting for redis queues to start...")
            time.sleep(1)
