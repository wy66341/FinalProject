"""JPL Horizons 代理补丁 — 课程提供.

对 astroquery.jplhorizons.Horizons 打补丁，将请求路由至课程代理服务器
http://8.216.49.176:18766/api/horizons.api

用法：在任何导入 astroquery 之前先 import jpl_forward。
"""

import urllib.request
import urllib.error

PROXY_URL = 'http://8.216.49.176:18766/api/horizons.api'
TOKEN = 'fce2a741a94feded5c3b9b7ab51f6748'


def _patch_horizons():
    """对 astroquery Horizons 做 URL 补丁."""
    try:
        from astroquery.jplhorizons import Horizons, conf

        # 替换 API URL
        conf.horizons_server = PROXY_URL

        # 添加认证头
        original_request = urllib.request.Request

        class PatchedRequest(original_request):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.add_header('Authorization', f'Bearer {TOKEN}')

        urllib.request.Request = PatchedRequest

        print('[jpl_forward] Horizons proxy patched.')
    except ImportError:
        print('[jpl_forward] astroquery not installed; skipping patch.')


_patch_horizons()
