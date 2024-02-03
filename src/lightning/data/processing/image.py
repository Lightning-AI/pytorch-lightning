import io
import urllib


def is_disallowed(headers, user_agent_token, disallowed_header_directives):
    """Check if HTTP headers contain an X-Robots-Tag directive disallowing usage."""
    for values in headers.get_all("X-Robots-Tag", []):
        try:
            uatoken_directives = values.split(":", 1)
            directives = [x.strip().lower() for x in uatoken_directives[-1].split(",")]
            ua_token = uatoken_directives[0].lower() if len(uatoken_directives) == 2 else None
            if (ua_token is None or ua_token == user_agent_token) and any(
                x in disallowed_header_directives for x in directives
            ):
                return True
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"Failed to parse X-Robots-Tag: {values}: {err}")
    return False


def download_image(url, timeout=10, user_agent_token="img2dataset", disallowed_header_directives=["noai", "noimageai", "noindex", "noimageindex"]):
    """Download an image with urllib."""
    url
    img_stream = None
    user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
    if user_agent_token:
        user_agent_string += f" (compatible; {user_agent_token}; +https://github.com/rom1504/img2dataset)"
    try:
        request = urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string})
        with urllib.request.urlopen(request, timeout=timeout) as r:
            if disallowed_header_directives and is_disallowed(
                r.headers,
                user_agent_token,
                disallowed_header_directives,
            ):
                return key, None, "Use of image disallowed by X-Robots-Tag directive"
            img_stream = io.BytesIO(r.read())
        return img_stream, None
    except Exception as err:  # pylint: disable=broad-except
        if img_stream is not None:
            img_stream.close()
        return None, str(err)


def download_image_with_retry(retries, url, timeout=10, user_agent_token="img2dataset", disallowed_header_directives=[]):
    for _ in range(retries + 1):
        img_stream, err = download_image(url, timeout, user_agent_token, disallowed_header_directives)
        if img_stream is not None:
            return img_stream, err
    return None, err
