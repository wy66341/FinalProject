"""JPL Horizons 接入补丁.

将 JPL 官方 API 请求重定向至课程代理服务器。
Token 作为 URL 查询参数传递（非 HTTP Header）。

用法：在任何导入 astroquery 之前先 import jpl_forward。
"""

import requests

PROXY_URL = 'http://8.216.49.176:18766/api/horizons.api'
TOKEN = 'fce2a741a94feded5c3b9b7ab51f6748'


def is_proxy_available():
    """检查课程代理是否可用."""
    try:
        resp = requests.get(PROXY_URL, timeout=8,
                            params={'token': TOKEN, 'format': 'text'})
        return resp.status_code == 200
    except Exception:
        return False


def _patch_horizons():
    """Smart patch: try proxy, redirect JPL requests with token in query."""
    try:
        import astroquery.jplhorizons  # noqa: F401
    except ImportError:
        print('[jpl_forward] astroquery not installed; skipping patch.')
        return

    if not is_proxy_available():
        print('[jpl_forward] WARNING: Course proxy unavailable (HTTP not 200).')
        print('[jpl_forward] Horizons queries will fail. Check token/proxy URL.')
        return

    _original = requests.Session.request

    def _patched(self, method, url, params=None, headers=None, **kwargs):
        if headers is None:
            headers = {}
        if 'ssd.jpl.nasa.gov' in url:
            url = PROXY_URL
            if params is None:
                params = {}
            # Token as query parameter (NOT Bearer header)
            params['token'] = TOKEN
        return _original(self, method, url, params=params, headers=headers, **kwargs)

    requests.Session.request = _patched
    print('[jpl_forward] JPL proxy active (token in query params).')


_patch_horizons()
