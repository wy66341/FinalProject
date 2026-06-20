"""JPL Horizons 接入补丁.

策略：优先直连 JPL 官方 API，不可用时走课程代理。
采用 requests 层 URL 重定向，不修改 astroquery 配置。

用法：在任何导入 astroquery 之前先 import jpl_forward。
"""

import requests

PROXY_URL = 'http://8.216.49.176:18766/api/horizons.api'
TOKEN = 'fce2a741a94feded5c3b9b7ab51f6748'
JPL_URL = 'https://ssd.jpl.nasa.gov/api/horizons.api'

_direct_available = None
_proxy_available = None


def is_direct_available():
    """检查是否能直连 JPL."""
    global _direct_available
    if _direct_available is not None:
        return _direct_available
    try:
        resp = requests.get(JPL_URL, timeout=8)
        _direct_available = resp.status_code < 500
    except Exception:
        _direct_available = False
    return _direct_available


def is_proxy_available():
    """检查课程代理是否可用."""
    global _proxy_available
    if _proxy_available is not None:
        return _proxy_available
    try:
        resp = requests.get(PROXY_URL, timeout=5,
                            headers={'Authorization': f'Bearer {TOKEN}'})
        _proxy_available = resp.status_code == 200
    except Exception:
        _proxy_available = False
    return _proxy_available


def _patch_horizons():
    """Smart patch: try direct first, fall back to proxy."""
    try:
        import astroquery.jplhorizons  # noqa: F401
    except ImportError:
        print('[jpl_forward] astroquery not installed; skipping patch.')
        return

    direct = is_direct_available()
    proxy = is_proxy_available() if not direct else False

    if direct:
        print('[jpl_forward] JPL direct access available — no proxy needed.')
        return

    if not proxy:
        print('[jpl_forward] WARNING: JPL direct access TIMEOUT, proxy 403.')
        print('[jpl_forward] All Horizons queries will fail. Contact TA for updated proxy.')
        return

    # Proxy mode: redirect JPL requests to proxy with auth
    _original = requests.Session.request

    def _patched(self, method, url, headers=None, **kwargs):
        if headers is None:
            headers = {}
        if 'ssd.jpl.nasa.gov' in url:
            url = PROXY_URL
            headers['Authorization'] = f'Bearer {TOKEN}'
        return _original(self, method, url, headers=headers, **kwargs)

    requests.Session.request = _patched
    print('[jpl_forward] Direct JPL blocked, using course proxy.')


_patch_horizons()
