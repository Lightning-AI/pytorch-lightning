"""Script for filtering unused caches."""

import os
from datetime import timedelta


def fetch_all_caches(repository: str, token: str, per_page: int = 100, max_pages: int = 100) -> list[dict]:
    """Fetch list of al caches from a given repository.

    Args:
        repository: user / repo-name
        token: authentication token for GH API calls
        per_page: number of items per listing page
        max_pages: max number of listing pages

    """
    import requests
    from pandas import Timestamp, to_datetime

    # Initialize variables for pagination
    all_caches = []

    for page in range(max_pages):
        # Get a page of caches for the repository
        url = f"https://api.github.com/repos/{repository}/actions/caches?page={page + 1}&per_page={per_page}"
        headers = {"Authorization": f"token {token}"}
        response = requests.get(url, headers=headers, timeout=10).json()
        if "total_count" not in response:
            raise RuntimeError(response.get("message"))
        print(f"fetching page... {page} with {per_page} items of expected {response.get('total_count')}")
        caches = response.get("actions_caches", [])

        # Append the caches from this page to the overall list
        all_caches.extend(caches)

        # Check if there are more pages to retrieve
        if len(caches) < per_page:
            break

    # Iterate through all caches and list them
    if all_caches:
        current_date = Timestamp.now(tz="UTC")
        print(f"Caches {len(all_caches)} for {repository}:")
        for cache in all_caches:
            cache_key = cache["id"]
            created_at = to_datetime(cache["created_at"])
            last_used_at = to_datetime(cache["last_accessed_at"])
            cache["last_used_days"] = current_date - last_used_at
            age_used = cache["last_used_days"].round(freq="min")
            size = cache["size_in_bytes"] / (1024 * 1024)
            print(
                f"- Cache Key: {cache_key} |"
                f" Created At: {created_at.strftime('%Y-%m-%d %H:%M')} |"
                f" Used At: {last_used_at.strftime('%Y-%m-%d %H:%M')} [{age_used}] |"
                f" Size: {size:.2f} MB"
            )
    else:
        print("No caches found for the repository.")
    return all_caches


def main(repository: str, token: str, age_days: float = 7, output_file: str = "unused-cashes.txt") -> None:
    """Entry point for CLI.

    Args:
        repository: GitHub repository name in form `<user>/<repo>`
        token: authentication token for making API calls
        age_days: filter all caches older than this age set in days
        output_file: path to a file for dumping list of cache's Id

    """
    caches = fetch_all_caches(repository, token)

    delta_days = timedelta(days=age_days)
    old_caches = [str(cache["id"]) for cache in caches if cache["last_used_days"] > delta_days]
    print(f"found {len(old_caches)} old caches:\n {old_caches}")

    with open(output_file, "w", encoding="utf8") as fw:
        fw.write(os.linesep.join(old_caches))


if __name__ == "__main__":
    from jsonargparse import auto_cli, set_parsing_settings

    set_parsing_settings(parse_optionals_as_positionals=True)
    auto_cli(main)
