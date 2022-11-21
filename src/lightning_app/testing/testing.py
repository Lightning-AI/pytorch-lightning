import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from multiprocessing import Process
from subprocess import Popen
from time import sleep
from typing import Any, Callable, Dict, Generator, List, Optional, Type

import requests
from lightning_cloud.openapi.rest import ApiException
from requests import Session
from rich import print
from rich.color import ANSI_COLOR_NAMES

from lightning_app import LightningApp, LightningFlow
from lightning_app.cli.lightning_cli import run_app
from lightning_app.core.constants import LIGHTNING_CLOUD_PROJECT_ID
from lightning_app.runners.multiprocess import MultiProcessRuntime
from lightning_app.testing.config import _Config
from lightning_app.utilities.app_logs import _app_logs_reader
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.enum import CacheCallsKeys
from lightning_app.utilities.imports import _is_playwright_available, requires
from lightning_app.utilities.log import get_logfile
from lightning_app.utilities.logs_socket_api import _LightningLogsSocketAPI
from lightning_app.utilities.network import _configure_session, LightningClient
from lightning_app.utilities.packaging.lightning_utils import get_dist_path_if_editable_install
from lightning_app.utilities.proxies import ProxyWorkRun

if _is_playwright_available():
    import playwright
    from playwright.sync_api import HttpCredentials, sync_playwright


def _on_error_callback(ws_app, *_):
    ws_app.close()


def _print_logs(app_id: str):
    client = LightningClient()
    project = _get_project(client)

    works = client.lightningwork_service_list_lightningwork(
        project_id=project.project_id,
        app_id=app_id,
    ).lightningworks
    component_names = ["flow"] + [w.name for w in works]

    rich_colors = list(ANSI_COLOR_NAMES)
    colors = {c: rich_colors[i + 1] for i, c in enumerate(component_names)}

    max_length = max(len(c.replace("root.", "")) for c in component_names)
    identifiers = []

    print("################### PRINTING LOGS ###################")

    logs_api_client = _LightningLogsSocketAPI(client.api_client)

    while True:
        gen = _app_logs_reader(
            logs_api_client=logs_api_client,
            project_id=project.project_id,
            app_id=app_id,
            component_names=component_names,
            follow=False,
            on_error_callback=_on_error_callback,
        )
        for log_event in gen:
            message = log_event.message
            identifier = f"{log_event.timestamp}{log_event.message}"
            if identifier not in identifiers:
                date = log_event.timestamp.strftime("%m/%d/%Y %H:%M:%S")
                identifiers.append(identifier)
                color = colors[log_event.component_name]
                padding = (max_length - len(log_event.component_name)) * " "
                print(f"[{color}]{log_event.component_name}{padding}[/{color}] {date} {message}")


class LightningTestApp(LightningApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    @staticmethod
    def _configure_session() -> Session:
        return _configure_session()

    def make_request(self, fn, *args, **kwargs):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self._make_request(fn, *args, **kwargs))

    async def _make_request(self, fn: Callable, *args, **kwargs):
        from lightning_app.utilities.state import AppState

        state = AppState()
        state._request_state()
        fn(state, *args, **kwargs)
        state.send_delta()

    def on_before_run_once(self):
        pass

    def on_after_run_once(self):
        self.counter += 1

    def run_once(self):
        before_done = self.on_before_run_once()
        if before_done is not None:
            return before_done
        done = super().run_once()
        after_done = self.on_after_run_once()
        if after_done is not None:
            return after_done
        return done

    def kill_work(self, work_name: str, sleep_time: int = 1):
        """Use this method to kill a specific work by its name."""
        self.processes[work_name].kill()

    def restart_work(self, work_name: str):
        """Use this method to restart a specific work by its name."""
        self.processes[work_name].restart()


@requires("click")
def application_testing(
    lightning_app_cls: Type[LightningTestApp] = LightningTestApp, command_line: List[str] = []
) -> Any:
    from unittest import mock

    from click.testing import CliRunner

    patch1 = mock.patch("lightning_app.LightningApp", lightning_app_cls)
    # we need to patch both only with the mirror package
    patch2 = (
        mock.patch("lightning.LightningApp", lightning_app_cls) if "lightning.app" in sys.modules else nullcontext()
    )
    with patch1, patch2:
        original = sys.argv
        sys.argv = command_line
        runner = CliRunner()
        result = runner.invoke(run_app, command_line, catch_exceptions=False)
        sys.argv = original
        return result


class _SingleWorkFlow(LightningFlow):
    def __init__(self, work, args, kwargs):
        super().__init__()
        self.work = work
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if self.work.has_succeeded or self.work.has_failed:
            self._exit()
        self.work.run(*self.args, **self.kwargs)


def run_work_isolated(work, *args, start_server: bool = False, **kwargs):
    """This function is used to run a work a single time with multiprocessing runtime."""
    MultiProcessRuntime(
        LightningApp(_SingleWorkFlow(work, args, kwargs), log_level="debug"),
        start_server=start_server,
    ).dispatch()
    # pop the stopped status.
    call_hash = work._calls[CacheCallsKeys.LATEST_CALL_HASH]

    if call_hash in work._calls:
        work._calls[call_hash]["statuses"].pop(-1)

    if isinstance(work.run, ProxyWorkRun):
        work.run = work.run.work_run


def _browser_context_args(browser_context_args: Dict) -> Dict:
    return {
        **browser_context_args,
        "viewport": {
            "width": 1920,
            "height": 1080,
        },
        "ignore_https_errors": True,
    }


@contextmanager
def _run_cli(args) -> Generator:
    """This utility is used to automate end-to-end testing of the Lightning AI CLI."""
    cmd = [
        sys.executable,
        "-m",
        "lightning",
    ] + args

    with tempfile.TemporaryDirectory() as tmpdir:
        env_copy = os.environ.copy()
        process = Popen(
            cmd,
            cwd=tmpdir,
            env=env_copy,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process.wait()

    yield process.stdout.read().decode("UTF-8"), process.stderr.read().decode("UTF-8")


@requires("playwright")
@contextmanager
def run_app_in_cloud(
    app_folder: str, app_name: str = "app.py", extra_args: List[str] = [], debug: bool = True
) -> Generator:
    """This utility is used to automate testing e2e application with lightning_app.ai."""
    # 1. Validate the provide app_folder is correct.
    if not os.path.exists(os.path.join(app_folder, "app.py")):
        raise Exception("The app folder should contain an app.py file.")
    if app_folder.endswith("/"):
        app_folder = app_folder[:-1]

    # 2. Create the right application name.
    basename = app_folder.split("/")[-1]
    PR_NUMBER = os.getenv("PR_NUMBER", None)

    is_editable_mode = get_dist_path_if_editable_install("lightning")
    if not is_editable_mode and PR_NUMBER is not None:
        raise Exception("Lightning requires to be installed in editable mode in the CI.")

    TEST_APP_NAME = os.getenv("TEST_APP_NAME", basename)
    os.environ["TEST_APP_NAME"] = TEST_APP_NAME

    if PR_NUMBER:
        name = f"test-{PR_NUMBER}-{TEST_APP_NAME}-" + str(int(time.time()))
    else:
        name = f"test-{TEST_APP_NAME}-" + str(int(time.time()))

    os.environ["LIGHTNING_APP_NAME"] = name

    url = _Config.url
    if url.endswith("/"):
        url = url[:-1]
    payload = {"apiKey": _Config.api_key, "username": _Config.username}
    res = requests.post(url + "/v1/auth/login", data=json.dumps(payload))
    if "token" not in res.json():
        raise Exception("You haven't properly setup your environment variables.")

    token = res.json()["token"]

    # 3. Disconnect from the App if any.
    Popen("lightning disconnect", shell=True).wait()

    # 4. Launch the application in the cloud from the Lightning CLI.
    with tempfile.TemporaryDirectory() as tmpdir:
        env_copy = os.environ.copy()
        env_copy["PACKAGE_LIGHTNING"] = "1"
        if debug:
            env_copy["LIGHTNING_DEBUG"] = "1"
        shutil.copytree(app_folder, tmpdir, dirs_exist_ok=True)
        # TODO - add -no-cache to the command line.
        stdout_path = get_logfile(f"run_app_in_cloud_{name}")
        with open(stdout_path, "w") as stdout:
            cmd = [
                sys.executable,
                "-m",
                "lightning",
                "run",
                "app",
                app_name,
                "--cloud",
                "--name",
                name,
                "--open-ui",
                "false",
            ]
            process = Popen((cmd + extra_args), cwd=tmpdir, env=env_copy, stdout=stdout, stderr=sys.stderr)
            process.wait()

        if is_editable_mode:
            # Added to ensure the current code is properly uploaded.
            # Otherwise, it could result in un-tested PRs.
            pkg_found = False
            with open(stdout_path) as fo:
                for line in fo.readlines():
                    if "Packaged Lightning with your application" in line:
                        pkg_found = True
                    print(line)  # TODO: use logging
            assert pkg_found
        os.remove(stdout_path)

    # 5. Print your application name
    print(f"The Lightning App Name is: [bold magenta]{name}[/bold magenta]")

    # 6. Create chromium browser, auth to lightning_app.ai and yield the admin and view pages.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=bool(int(os.getenv("HEADLESS", "0"))))
        context = browser.new_context(
            # Eventually this will need to be deleted
            http_credentials=HttpCredentials(
                {"username": os.getenv("LAI_USER", "").strip(), "password": os.getenv("LAI_PASS", "")}
            ),
            record_video_dir=os.path.join(_Config.video_location, TEST_APP_NAME),
            record_har_path=_Config.har_location,
        )
        admin_page = context.new_page()
        print(f"The Lightning App Token is: {token}")
        print(f"The Lightning App user key is: {_Config.key}")
        print(f"The Lightning App user id is: {_Config.id}")
        admin_page.goto(_Config.url)
        admin_page.evaluate(
            """data => {
            window.localStorage.setItem('gridUserId', data[0]);
            window.localStorage.setItem('gridUserKey', data[1]);
            window.localStorage.setItem('gridUserToken', data[2]);
        }
        """,
            [_Config.id, _Config.key, token],
        )
        if LIGHTNING_CLOUD_PROJECT_ID:
            admin_page.evaluate(
                """data => {
                window.localStorage.setItem('gridDefaultProjectIdOverride', JSON.stringify(data[0]));
            }
            """,
                [LIGHTNING_CLOUD_PROJECT_ID],
            )
        admin_page.goto(f"{_Config.url}/{_Config.username}/apps", timeout=60 * 1000)

        # Closing the Complete your profile dialog
        try:
            dialog = admin_page.locator("text=Complete your profile")
            dialog.wait_for(timeout=10 * 1000, state="visible")
            print("'Complete your profile' dialog visible, closing it.")
            admin_page.locator('input[name="firstName"]').fill("first")
            admin_page.locator('input[name="lastName"]').fill("last")
            admin_page.locator('input[name="email"]').fill("e2e.test.admin@lightning.ai")
            admin_page.locator('input[name="organization"]').fill("Lightning AI")
            button = admin_page.locator('button:has-text("Confirm")')
            button.wait_for(timeout=3 * 1000)
            button.click()
        except playwright._impl._api_types.TimeoutError:
            print("'Complete your profile' dialog not visible, skipping.")

        # Closing the Create Project dialog.
        try:
            project_dialog = admin_page.locator("text=Create a project")
            project_dialog.wait_for(timeout=10 * 1000, state="visible")
            print("'Create Project' dialog visible, closing it.")
            project_name_input = admin_page.locator('input[type="text"]')
            project_name_input.fill("Default Project")
            button = admin_page.locator('button:has-text("Continue")')
            button.wait_for(timeout=3 * 1000)
            button.click()
        except playwright._impl._api_types.TimeoutError:
            print("'Create Project' dialog not visible, skipping.")

        admin_page.locator(f"role=link[name='{name}']").click()
        sleep(5)
        # Scroll to the bottom of the page. Used to capture all logs.
        admin_page.evaluate(
            """
            var intervalID = setInterval(function () {
                var scrollingElement = (document.scrollingElement || document.body);
                scrollingElement.scrollTop = scrollingElement.scrollHeight;
            }, 200);
            """
        )

        client = LightningClient()
        project = _get_project(client)

        lit_apps = [
            app
            for app in client.lightningapp_instance_service_list_lightningapp_instances(
                project_id=project.project_id
            ).lightningapps
            if app.name == name
        ]

        if not lit_apps:
            return True

        assert len(lit_apps) == 1
        app_id = lit_apps[0].id

        if debug:
            process = Process(target=_print_logs, kwargs={"app_id": app_id})
            process.start()

        while True:
            try:
                with admin_page.context.expect_page() as page_catcher:
                    admin_page.locator('[data-cy="open"]').click()
                view_page = page_catcher.value
                view_page.wait_for_load_state(timeout=0)
                break
            except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError):
                pass

        lit_apps = [
            app
            for app in client.lightningapp_instance_service_list_lightningapp_instances(
                project_id=project.project_id
            ).lightningapps
            if app.name == name
        ]

        app_url = lit_apps[0].status.url

        while True:
            sleep(1)
            resp = requests.get(app_url + "/openapi.json")
            if resp.status_code == 200:
                break

        print(f"The Lightning Id Name : [bold magenta]{app_id}[/bold magenta]")

        logs_api_client = _LightningLogsSocketAPI(client.api_client)

        def fetch_logs(component_names: Optional[List[str]] = None) -> Generator:
            """This methods creates websockets connection in threads and returns the logs to the main thread."""
            if not component_names:
                works = client.lightningwork_service_list_lightningwork(
                    project_id=project.project_id,
                    app_id=app_id,
                ).lightningworks

                component_names = ["flow"] + [w.name for w in works]
            else:

                def add_prefix(c: str) -> str:
                    if c == "flow":
                        return c
                    if not c.startswith("root."):
                        return "root." + c
                    return c

                component_names = [add_prefix(c) for c in component_names]

            gen = _app_logs_reader(
                logs_api_client=logs_api_client,
                project_id=project.project_id,
                app_id=app_id,
                component_names=component_names,
                follow=False,
                on_error_callback=_on_error_callback,
            )
            for log_event in gen:
                yield log_event.message

        try:
            yield admin_page, view_page, fetch_logs, name
        except KeyboardInterrupt:
            pass
        finally:
            if debug:
                process.kill()

            context.close()
            browser.close()
            Popen("lightning disconnect", shell=True).wait()


def wait_for(page, callback: Callable, *args, **kwargs) -> Any:
    import playwright

    while True:
        try:
            res = callback(*args, **kwargs)
            if res:
                return res
        except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError) as e:
            print(e)
            try:
                sleep(5)
                page.reload()
            except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError) as e:
                print(e)
                pass
            sleep(2)


def _delete_lightning_app(client, project_id, app_id, app_name):
    print(f"Deleting {app_name} id: {app_id}")
    try:
        res = client.lightningapp_instance_service_delete_lightningapp_instance(
            project_id=project_id,
            id=app_id,
        )
        assert res == {}
    except ApiException as ex:
        print(f"Failed to delete {app_name}. Exception {ex}")


def delete_cloud_lightning_apps():
    """Cleanup cloud apps that start with the name test-{PR_NUMBER}-{TEST_APP_NAME}.

    PR_NUMBER and TEST_APP_NAME are environment variables.
    """

    client = LightningClient()

    try:
        pr_number = int(os.getenv("PR_NUMBER", None))
    except (TypeError, ValueError):
        # Failed when the PR is running master or 'PR_NUMBER' isn't defined.
        pr_number = ""

    app_name = os.getenv("TEST_APP_NAME", "")

    print(f"deleting apps for pr_number: {pr_number}, app_name: {app_name}")
    project = _get_project(client)
    list_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project.project_id)

    for lit_app in list_apps.lightningapps:
        if pr_number and app_name and not lit_app.name.startswith(f"test-{pr_number}-{app_name}-"):
            continue
        _delete_lightning_app(client, project_id=project.project_id, app_id=lit_app.id, app_name=lit_app.name)

    print("deleting apps that were created more than 1 hour ago.")

    for lit_app in list_apps.lightningapps:

        if lit_app.created_at < datetime.datetime.now(lit_app.created_at.tzinfo) - datetime.timedelta(hours=1):
            _delete_lightning_app(client, project_id=project.project_id, app_id=lit_app.id, app_name=lit_app.name)
