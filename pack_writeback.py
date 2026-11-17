"""Write a stock Director ZIP back to NVStudio output without running ComfyUI."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


OUTPUT_ROOT = Path(r"F:\NVStudio-output")
PACK_NAME = re.compile(r"(?P<project>[\w]+)_第(?P<episode>[0-9]+)集_beat(?P<beat>[0-9]{3})\.zip")


def pack_filename(project, episode, beat):
    if not isinstance(project, str) or not re.fullmatch(r"\w+", project):
        raise ValueError("项目名只能包含中文、字母、数字和下划线")
    if type(episode) is not int or episode < 0:
        raise ValueError("集数必须是非负整数")
    if type(beat) is not int or not 0 <= beat <= 999:
        raise ValueError("拍号必须是 0–999 的整数（三位数）")
    return f"{project}_第{episode}集_beat{beat:03d}.zip"


def parse_pack_filename(name):
    match = PACK_NAME.fullmatch(name)
    if match is None:
        return None
    return match['project'], int(match['episode']), int(match['beat'])


def _output_path(path, root):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError(f"output 原路径必须是绝对路径: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"只允许写到 {root} 下: {path}")
    if resolved.suffix.lower() != '.zip':
        raise ValueError(f"output 路径必须是 ZIP 文件: {path}")
    return resolved


def _unique_match(name, root):
    matches = [p for p in root.rglob('*') if p.is_file() and p.name.casefold() == name.casefold()]
    if len(matches) > 1:
        raise ValueError(f"output 中有多个同名包，请提供 --original 完整路径: {name}")
    if not matches:
        raise ValueError(f"output 中没有唯一对应包，请提供 --original 完整路径: {name}")
    return _output_path(matches[0], root)


def resolve_destination(source, original=None, project=None, episode=None, beat=None, *, output_root=OUTPUT_ROOT):
    root = Path(output_root).resolve()
    source = Path(source).resolve()
    source_info = parse_pack_filename(source.name)
    if original:
        original = Path(original)
        if original.name == str(original):
            anchor = _unique_match(original.name, root)
        else:
            anchor = _output_path(original, root)
    elif source_info:
        p, e, _ = source_info
        anchor = root / p / f"第{e}集" / source.name
    elif source.is_relative_to(root):
        anchor = _output_path(source, root)
    else:
        anchor = _unique_match(source.name, root)
    anchor = _output_path(anchor, root)
    info = parse_pack_filename(anchor.name)
    if info:
        # An explicit original filename owns its identity, including the beat number.
        return anchor, None
    folder_episode = re.fullmatch(r"第([0-9]+)集", anchor.parent.name)
    inferred_project = anchor.parent.parent.name if folder_episode else None
    inferred_episode = int(folder_episode[1]) if folder_episode else None
    project = project if project is not None else inferred_project or (source_info[0] if source_info else None)
    episode = episode if episode is not None else inferred_episode if inferred_episode is not None else (source_info[1] if source_info else None)
    beat = beat if beat is not None else (source_info[2] if source_info else None)
    if project is None or episode is None or beat is None:
        raise ValueError("无法解析项目名、集数或拍号；请补充 --project、--episode、--beat")
    target = _output_path(anchor.with_name(pack_filename(project, episode, beat)), root)
    return target, anchor if anchor != target else None


def replace_pack(source, target):
    """Stage and validate before moving the original; restore it on replace failure."""
    temporary = None
    backup = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.pack-writeback-', suffix='.tmp', delete=False) as staged:
            temporary = Path(staged.name)
            with source.open('rb') as incoming:
                shutil.copyfileobj(incoming, staged)
            staged.flush()
            os.fsync(staged.fileno())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip():
                raise ValueError("ZIP 校验失败")
        if target.exists():
            backup = target.with_name(target.name + datetime.now().strftime('.bak-%Y%m%d-%H%M%S'))
            if backup.exists():
                raise FileExistsError(f"备份已存在，请稍后重试: {backup}")
            target.rename(backup)
        try:
            os.replace(temporary, target)
        except OSError:
            if backup is not None:
                backup.rename(target)
            raise
        return backup
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def writeback_pack(source, original=None, project=None, episode=None, beat=None, *, output_root=OUTPUT_ROOT):
    source = Path(source).resolve(strict=True)
    target, old = resolve_destination(source, original, project, episode, beat, output_root=output_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    replace_pack(source, target)
    if old is not None and old.exists():
        try:
            old.unlink()
        except OSError as exc:
            raise OSError(f"新包已写入 {target}，但旧包未能删除 {old}: {exc}") from exc
    return target


def main():
    parser = argparse.ArgumentParser(description='将 Director 下载的 ZIP 写回 F:\\NVStudio-output；不启动 ComfyUI')
    parser.add_argument('source', help='刚导出的 ZIP 完整路径（Downloads 中的原文件无需改名）')
    parser.add_argument('--original', help='output 原包完整路径；也可提供能唯一匹配的原文件名')
    parser.add_argument('--project', help='无法自动解析时提供项目名')
    parser.add_argument('--episode', type=int, help='无法自动解析时提供集数')
    parser.add_argument('--beat', type=int, help='旧名包对应的拍号，例如 3')
    args = parser.parse_args()
    try:
        result = writeback_pack(args.source, args.original, args.project, args.episode, args.beat)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        parser.exit(1, f"写回失败: {exc}\n")
    print(f"写回成功: {result}")


if __name__ == '__main__':
    main()
