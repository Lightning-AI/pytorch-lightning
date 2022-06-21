import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from subprocess import Popen
from time import sleep
from typing import Any, Callable, Dict, Generator, List, Type

import requests
from lightning_cloud.openapi.rest import ApiException
from requests import Session
from rich import print

from lightning_app import LightningApp, LightningFlow
from lightning_app.cli.lightning_cli import run_app
from lightning_app.core.constants import LIGHTNING_CLOUD_PROJECT_ID
from lightning_app.runners.multiprocess import MultiProcessRuntime
from lightning_app.testing.config import Config
from lightning_app.utilities.cloud import _get_project
from lightning_app.utilities.imports import _is_playwright_available, requires
from lightning_app.utilities.network import _configure_session, LightningClient

if _is_playwright_available():
    import playwright
    from playwright.sync_api import HttpCredentials, sync_playwright


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
def application_testing(lightning_app_cls: Type[LightningTestApp], command_line: List[str] = []) -> Any:
    from unittest import mock

    from click.testing import CliRunner

    with mock.patch("lightning_app.LightningApp", lightning_app_cls):
        runner = CliRunner()
        return runner.invoke(run_app, command_line, catch_exceptions=False)


class SingleWorkFlow(LightningFlow):
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
        LightningApp(SingleWorkFlow(work, args, kwargs), debug=True),
        start_server=start_server,
    ).dispatch()
    # pop the stopped status.
    call_hash = work._calls["latest_call_hash"]
    work._calls[call_hash]["statuses"].pop(-1)


def browser_context_args(browser_context_args: Dict) -> Dict:
    return {
        **browser_context_args,
        "viewport": {
            "width": 1920,
            "height": 1080,
        },
        "ignore_https_errors": True,
    }


@requires("playwright")
@contextmanager
def run_app_in_cloud(app_folder: str, app_name: str = "app.py") -> Generator:
    """This utility is used to automate testing e2e application with lightning_app.ai."""
    # 1. Validate the provide app_folder is correct.
    if not os.path.exists(os.path.join(app_folder, "app.py")):
        raise Exception("The app folder should contain an app.py file.")
    if app_folder.endswith("/"):
        app_folder = app_folder[:-1]

    # 2. Create the right application name.
    basename = app_folder.split("/")[-1]
    PR_NUMBER = os.getenv("PR_NUMBER", None)
    TEST_APP_NAME = os.getenv("TEST_APP_NAME", basename)
    if PR_NUMBER:
        name = f"test-{PR_NUMBER}-{TEST_APP_NAME}-" + str(int(time.time()))
    else:
        name = f"test-{TEST_APP_NAME}-" + str(int(time.time()))

    # 3. Launch the application in the cloud from the Lightning CLI.
    with tempfile.TemporaryDirectory() as tmpdir:
        env_copy = os.environ.copy()
        env_copy["PREPARE_LIGHTING"] = "1"
        shutil.copytree(app_folder, tmpdir, dirs_exist_ok=True)
        # TODO - add -no-cache to the command line.
        process = Popen(
            [
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
            ],
            cwd=tmpdir,
            env=env_copy,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        process.wait()

    # 4. Print your application name
    print(f"The Lightning App Name is: [bold magenta]{name}[/bold magenta]")

    # 5. Create chromium browser, auth to lightning_app.ai and yield the admin and view pages.
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=bool(int(os.getenv("HEADLESS", "0"))))
        payload = {
            "apiKey": Config.api_key,
            "username": Config.username,
            "duration": "120000",
        }
        context = browser.new_context(
            # Eventually this will need to be deleted
            http_credentials=HttpCredentials({"username": os.getenv("LAI_USER"), "password": os.getenv("LAI_PASS")}),
            record_video_dir=os.path.join(Config.video_location, TEST_APP_NAME),
            record_har_path=Config.har_location,
        )
        admin_page = context.new_page()
        res = requests.post(Config.url + "/v1/auth/login", data=json.dumps(payload))
        token = res.json()["token"]
        print(f"The Lightning App Token is: {token}")
        print(f"The Lightning App user key is: {Config.key}")
        print(f"The Lightning App user id is: {Config.id}")
        admin_page.goto(Config.url)
        admin_page.evaluate(
            """data => {
            window.localStorage.setItem('gridUserId', data[0]);
            window.localStorage.setItem('gridUserKey', data[1]);
            window.localStorage.setItem('gridUserToken', data[2]);
        }
        """,
            [Config.id, Config.key, token],
        )
        if LIGHTNING_CLOUD_PROJECT_ID:
            admin_page.evaluate(
                """data => {
                window.localStorage.setItem('gridDefaultProjectIdOverride', JSON.stringify(data[0]));
            }
            """,
                [LIGHTNING_CLOUD_PROJECT_ID],
            )
        admin_page.reload()
        try:
            # Closing the Create Project modal
            button = admin_page.locator('button:has-text("Cancel")')
            button.wait_for(timeout=1 * 1000)
            button.click()
        except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError):
            pass
        try:
            # Skipping the Hubspot form
            button = admin_page.locator('button:has-text("Skip for now")')
            button.wait_for(timeout=1 * 1000)
            button.click()
        except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError):
            pass
        admin_page.goto(f"{Config.url}/{Config.username}/apps")
        admin_page.locator(f"text={name}").click()
        admin_page.evaluate(
            """data => {
            window.localStorage.setItem('gridUserId', data[0]);
            window.localStorage.setItem('gridUserKey', data[1]);
            window.localStorage.setItem('gridUserToken', data[2]);
        }
        """,
            [Config.id, Config.key, token],
        )
        sleep(5)
        # Scroll to the bottom of the page. Used to capture all logs.
        admin_page.evaluate(
            """
            var intervalID = setInterval(function () {
                var scrollingElement = (document.scrollingElement || document.body);
                scrollingElement.scrollTop = scrollingElement.scrollHeight;
            }, 200);

            if (!window._logs) {
                window._logs = [];
            }

            if (window.logTerminals) {
                Object.entries(window.logTerminals).forEach(
                    ([key, value]) => {
                        window.logTerminals[key]._onLightningWritelnHandler = function (data) {
                            window._logs = window._logs.concat([data]);
                        }
                    }
                );
            }
            """
        )

        while True:
            try:
                with admin_page.context.expect_page() as page_catcher:
                    admin_page.locator('[data-cy="open"]').click()
                view_page = page_catcher.value
                view_page.wait_for_load_state(timeout=0)
                break
            except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError):
                pass

        def fetch_logs() -> str:
            return admin_page.evaluate("window._logs;")

        # 5. Print your application ID
        print(
            f"The Lightning Id Name : [bold magenta]{str(view_page.url).split('.')[0].split('//')[-1]}[/bold magenta]"
        )

        try:
            yield admin_page, view_page, fetch_logs
        except KeyboardInterrupt:
            pass
        finally:
            print("##################################################")
            printed_logs = []
            for log in fetch_logs():
                if log not in printed_logs:
                    printed_logs.append(log)
                    print(log.split("[0m")[-1])
            button = admin_page.locator('[data-cy="stop"]')
            try:
                button.wait_for(timeout=3 * 1000)
                button.click()
            except (playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError):
                pass
            context.close()
            browser.close()

            client = LightningClient()
            project = _get_project(client)
            list_lightningapps = client.lightningapp_instance_service_list_lightningapp_instances(project.project_id)

            for lightningapp in list_lightningapps.lightningapps:
                if lightningapp.name != name:
                    continue
                try:
                    res = client.lightningapp_instance_service_delete_lightningapp_instance(
                        project_id=project.project_id,
                        id=lightningapp.id,
                    )
                    assert res == {}
                except ApiException as e:
                    print(f"Failed to delete {lightningapp.name}. Exception {e}")


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
