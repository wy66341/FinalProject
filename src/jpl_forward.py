"""JPL Horizons 代理补丁 — 课程提供.

将 JPL 官方 API 请求重定向至课程代理服务器，并添加 Bearer 认证。
采用 requests 层 URL 重定向策略，不修改 astroquery 配置，避免新版本 astropy
的 ConfigItem 验证器拒绝非官方 URL。

用法：在任何导入 astroquery 之前先 import jpl_forward。
"""

"""JPL Horizons 代理补丁 — 课程提供.

将 JPL 官方 API 请求重定向至课程代理服务器，并添加 Bearer 认证。
采用 requests 层 URL 重定向策略，不修改 astroquery 配置，避免新版本 astropy
的 ConfigItem 验证器拒绝非官方 URL。

用法：在任何导入 astroquery 之前先 import jpl_forward。
"""

import requests

PROXY_URL = 'http://8.216.49.176:18766/api/horizons.api'
TOKEN = 'fce2a741a94feded5c3b9b7ab51f6748'

_proxy_available = None  # None = unknown, True = working, False = failed


def is_proxy_available():
    """检查代理服务器是否可用."""
    global _proxy_available
    if _proxy_available is not None:
        return _proxy_available
    try:
        resp = requests.get(PROXY_URL, timeout=5,
                            headers={'Authorization': f'Bearer {TOKEN}'})
        _proxy_available = resp.status_code != 403
    except Exception:
        _proxy_available = False
    return _proxy_available


def _patch_horizons():
    """在 requests 层面将 JPL 官方 URL 重定向到代理，并注入认证头."""
    try:
        import astroquery.jplhorizons  # noqa: F401

        _original_session_request = requests.Session.request

        def _patched_request(self, method, url, headers=None, **kwargs):
            if headers is None:
                headers = {}
            if 'ssd.jpl.nasa.gov' in url:
                url = PROXY_URL
                headers['Authorization'] = f'Bearer {TOKEN}'
            return _original_session_request(
                self, method, url, headers=headers, **kwargs
            )

        requests.Session.request = _patched_request
        print('[jpl_forward] Horizons proxy patched (URL redirect + Bearer auth).')

    except ImportError:
        print('[jpl_forward] astroquery not installed; skipping patch.')


_patch_horizons()
