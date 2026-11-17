from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pack_writeback as wb


class PackWritebackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'output'
        self.folder = self.root / '乡村猎艳1' / '第10集'
        self.folder.mkdir(parents=True)
        self.name = '乡村猎艳1_第10集_beat003.zip'
        self.source = self.make_zip(self.base / 'Downloads' / 'export-id.mmxpack.zip', 'new')

    def make_zip(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('manifest.json', value)
        return path

    def write(self, source=None, **kwargs):
        return wb.writeback_pack(source or self.source, output_root=self.root, **kwargs)

    def test_name_roundtrip(self):
        self.assertEqual(wb.pack_filename('乡村猎艳1', 10, 3), self.name)
        self.assertEqual(wb.parse_pack_filename(self.name), ('乡村猎艳1', 10, 3))
        self.assertIsNone(wb.parse_pack_filename('uuid.mmxpack.zip'))
        for project in ('../escape', 'a/b', 'a\\b', 'a:stream'):
            with self.subTest(project=project), self.assertRaises(ValueError):
                wb.pack_filename(project, 10, 3)

    def test_original_overwritten_source_preserved(self):
        target = self.make_zip(self.folder / self.name, 'old')
        original_data = target.read_bytes()
        data = self.source.read_bytes()
        self.assertEqual(self.write(original=target), target)
        self.assertEqual(target.read_bytes(), data)
        self.assertEqual(self.source.read_bytes(), data)
        backups = list(self.folder.glob(self.name + '.bak-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original_data)

    def test_original_priority_over_source_name_and_options(self):
        source = self.make_zip(self.base / '别的_第2集_beat099.zip', 'new')
        target = self.folder / self.name
        self.assertEqual(self.write(source, original=target, project='other', episode=1, beat=1), target)

    def test_conventional_filename_restores_directory(self):
        source = self.make_zip(self.base / self.name, 'new')
        self.assertEqual(self.write(source), self.folder / self.name)
        self.assertTrue(source.exists())

    def test_legacy_output_source_renamed_after_success(self):
        for name in ('550e8400-e29b-41d4-a716-446655440000.zip', 'export_id.mmxpack.zip'):
            with self.subTest(name=name):
                old = self.make_zip(self.folder / name, 'new')
                data = old.read_bytes()
                target = self.write(old, beat=3)
                self.assertEqual(target, self.folder / self.name)
                self.assertEqual(target.read_bytes(), data)
                self.assertFalse(old.exists())

    def test_download_replaces_legacy_original(self):
        old = self.make_zip(self.folder / 'uuid.zip', 'old')
        target = self.write(original=old, beat=3)
        self.assertFalse(old.exists())
        self.assertTrue(self.source.exists())
        self.assertEqual(target.read_bytes(), self.source.read_bytes())

    def test_filename_unique_match(self):
        old = self.make_zip(self.folder / self.source.name, 'old')
        self.assertEqual(self.write(beat=3), self.folder / self.name)
        self.assertFalse(old.exists())
        old = self.make_zip(self.folder / 'uuid.zip', 'old')
        self.assertEqual(self.write(original='uuid.zip', beat=3), self.folder / self.name)
        self.assertFalse(old.exists())

    def test_ambiguous_match_does_not_guess(self):
        self.make_zip(self.folder / self.source.name, 'old')
        self.make_zip(self.root / 'another' / self.source.name, 'old')
        with self.assertRaisesRegex(ValueError, '多个同名'):
            self.write(beat=3)
        self.assertTrue(self.source.exists())

    def test_missing_identity_does_not_guess(self):
        with self.assertRaisesRegex(ValueError, '没有唯一'):
            self.write()
        old = self.make_zip(self.folder / 'uuid.zip', 'old')
        with self.assertRaisesRegex(ValueError, '无法解析'):
            self.write(original=old)
        self.assertTrue(old.exists())

    def test_outside_and_traversal_rejected(self):
        for target in (self.base / self.name, self.root / '..' / self.name,
                       self.base / 'output-other' / self.name):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, '只允许'):
                self.write(original=target)
        self.assertTrue(self.source.exists())

    def test_symlink_escape_rejected(self):
        outside = self.base / 'outside'
        outside.mkdir()
        link = self.root / 'linked'
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('Windows symlink privilege unavailable')
        with self.assertRaisesRegex(ValueError, '只允许'):
            self.write(original=link / self.name)
        self.assertFalse((outside / self.name).exists())

    def test_failed_replace_keeps_both_original_and_source(self):
        old = self.make_zip(self.folder / 'uuid.zip', 'old')
        target = self.make_zip(self.folder / self.name, 'existing')
        old_data, target_data = old.read_bytes(), target.read_bytes()
        with patch.object(wb.os, 'replace', side_effect=OSError('locked')):
            with self.assertRaisesRegex(OSError, 'locked'):
                self.write(original=old, beat=3)
        self.assertEqual(old.read_bytes(), old_data)
        self.assertEqual(target.read_bytes(), target_data)
        self.assertTrue(self.source.exists())
        self.assertFalse(list(self.folder.glob('.pack-writeback-*')))

    def test_invalid_zip_keeps_original_and_source(self):
        old = self.make_zip(self.folder / 'uuid.zip', 'old')
        self.source.write_bytes(b'incomplete download')
        with self.assertRaises(zipfile.BadZipFile):
            self.write(original=old, beat=3)
        self.assertTrue(old.exists())
        self.assertTrue(self.source.exists())
        self.assertFalse((self.folder / self.name).exists())

    def test_same_source_and_target(self):
        target = self.make_zip(self.folder / self.name, 'new')
        data = target.read_bytes()
        self.assertEqual(self.write(target, original=target), target)
        self.assertEqual(target.read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
