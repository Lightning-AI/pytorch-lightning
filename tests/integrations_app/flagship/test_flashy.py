import contextlib
from time import sleep

import pytest
from lightning.app.testing.testing import run_app_in_cloud
from lightning.app.utilities.imports import _is_playwright_available

from integrations_app.flagship import _PATH_INTEGRATIONS_DIR

if _is_playwright_available():
    import playwright
    from playwright.sync_api import Page, expect


# TODO: when this function is moved to the app itself we can just import it, so to keep better aligned
def validate_app_functionalities(app_page: "Page") -> None:
    """Validate the page after app starts.

    this is direct copy-paste of validation living in the app repository:
    https://github.com/Lightning-AI/LAI-Flashy-App/blob/main/tests/test_app_gallery.py#L205

    app_page: The UI page of the app to be validated.

    """
    while True:
        with contextlib.suppress(playwright._impl._api_types.Error, playwright._impl._api_types.TimeoutError):
            app_page.reload()
            sleep(5)
            app_label = app_page.frame_locator("iframe").locator("text=Choose your AI task")
            app_label.wait_for(timeout=30 * 1000)
            break

    input_field = app_page.frame_locator("iframe").locator('input:below(:text("Data URL"))').first
    input_field.wait_for(timeout=1000)
    input_field.type("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip")
    sleep(1)
    upload_btn = app_page.frame_locator("iframe").locator('button:has-text("Upload")')
    upload_btn.wait_for(timeout=1000)
    upload_btn.click()

    sleep(10)

    train_folder_dropdown = app_page.frame_locator("iframe").locator("#mui-2")
    train_folder_dropdown.click()

    train_folder = app_page.frame_locator("iframe").locator('text="hymenoptera_data/train"')
    train_folder.scroll_into_view_if_needed()
    train_folder.click()

    val_folder_dropdown = app_page.frame_locator("iframe").locator("#mui-3")
    val_folder_dropdown.click()

    val_folder = app_page.frame_locator("iframe").locator('text="hymenoptera_data/val"')
    val_folder.scroll_into_view_if_needed()
    val_folder.click()

    train_btn = app_page.frame_locator("iframe").locator('button:has-text("Start training!")')
    train_btn.click()

    # Sometimes the results don't show until we refresh the page
    sleep(10)

    app_page.reload()

    app_page.frame_locator("iframe").locator('button:has-text("RESULTS")').click()
    runs = app_page.frame_locator("iframe").locator("table tbody tr")
    expect(runs).to_have_count(1, timeout=120000)


@pytest.mark.cloud()
def test_app_cloud() -> None:
    with run_app_in_cloud(_PATH_INTEGRATIONS_DIR) as (_, view_page, _, _):
        validate_app_functionalities(view_page)
