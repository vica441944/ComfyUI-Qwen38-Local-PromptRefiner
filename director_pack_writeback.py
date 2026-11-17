"""Local Director export writeback, with signed, browser-owned import identities."""

import asyncio
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import tempfile
import threading
import zipfile

if __package__:
    from .pack_writeback import OUTPUT_ROOT, _output_path, _unique_match, parse_pack_filename, replace_pack
else:
    from pack_writeback import OUTPUT_ROOT, _output_path, _unique_match, parse_pack_filename, replace_pack


class DirectorWriteback:
    def __init__(self, output_root=OUTPUT_ROOT):
        self.root = Path(output_root).resolve()
        self.key = secrets.token_bytes(32)
        self.lock = threading.Lock()

    def _signature(self, identity):
        data = json.dumps(identity, sort_keys=True, ensure_ascii=True).encode('utf-8')
        return hmac.new(self.key, data, hashlib.sha256).hexdigest()

    def remember(self, name):
        info = parse_pack_filename(name)
        if info is None:
            raise ValueError('无法确定原包，请从 output 用约定文件名再导入一次')
        target = _unique_match(name, self.root)
        identity = dict(path=str(target), project=info[0], episode=info[1], beat=info[2])
        self._target(identity)
        return dict(identity=identity, signature=self._signature(identity))

    def _target(self, identity):
        target = _output_path(identity['path'], self.root)
        info = parse_pack_filename(target.name)
        if info != (identity['project'], identity['episode'], identity['beat']):
            raise ValueError('原包项目、集数或拍号不一致，拒绝写回')
        if target.parent.name != f"第{info[1]}集" or target.parent.parent.name != info[0]:
            raise ValueError('原包目录与项目、集数不一致，拒绝写回')
        if _unique_match(target.name, self.root) != target:
            raise ValueError('原包路径不一致，拒绝写回')
        return target

    def write(self, source, ticket):
        if not isinstance(ticket, dict) or not isinstance(ticket.get('identity'), dict):
            raise ValueError('无有效导入身份，请从 output 用约定文件名再导入一次')
        identity = ticket['identity']
        if not isinstance(ticket.get('signature'), str) or not hmac.compare_digest(ticket['signature'], self._signature(identity)):
            raise ValueError('导入身份已失效或被更改，请重新导入原包')
        with self.lock:
            target = self._target(identity)
            with zipfile.ZipFile(source) as archive:
                if archive.testzip():
                    raise ValueError('ZIP 校验失败')
                if not {'pack.json', 'timeline.json'}.issubset(archive.namelist()):
                    raise ValueError('不是导演包：缺少 pack.json / timeline.json')
                try:
                    metadata = json.loads(archive.read('pack.json'))
                    timeline = json.loads(archive.read('timeline.json'))
                except (ValueError, UnicodeError):
                    raise ValueError('导演包 JSON 无效') from None
                if not isinstance(metadata, dict) or metadata.get('format') != 'minimax-h3-director-pack' or not isinstance(timeline, dict):
                    raise ValueError('不是 minimax-h3-director-pack 导演包')
            backup = replace_pack(Path(source), target)
            return dict(path=str(target), backup=str(backup) if backup else None)


_ROUTES_REGISTERED = False


def register_routes():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True
    try:
        from server import PromptServer
        from aiohttp import web
    except ImportError:
        return False
    server = getattr(PromptServer, 'instance', None)
    if server is None or not hasattr(server, 'routes'):
        return False
    service = DirectorWriteback()

    @server.routes.post('/qwen38/promptrefiner/director_import_identity')
    async def remember(request):
        try:
            body = await request.json()
            ticket = await asyncio.to_thread(service.remember, body.get('name', ''))
            return web.json_response(ticket)
        except (ValueError, OSError, TypeError):
            return web.json_response({'error': '无法唯一确定 output 原包，请用约定文件名重新导入'}, status=400)

    @server.routes.post('/qwen38/promptrefiner/director_export_writeback')
    async def write(request):
        try:
            with tempfile.TemporaryDirectory(prefix='qwen38_director_writeback_') as temporary:
                source = Path(temporary) / 'export.zip'
                ticket = None
                reader = await request.multipart()
                async for part in reader:
                    if part.name == 'identity':
                        ticket = json.loads(await part.text())
                    elif part.name == 'pack':
                        with source.open('wb') as output:
                            while chunk := await part.read_chunk():
                                output.write(chunk)
                    else:
                        raise ValueError('未知写回字段')
                result = await asyncio.to_thread(service.write, source, ticket)
                return web.json_response(result)
        except (ValueError, OSError, TypeError, KeyError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as error:
            return web.json_response({'error': str(error)}, status=400)

    _ROUTES_REGISTERED = True
    return True
