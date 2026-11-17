import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from director_pack_writeback import DirectorWriteback, register_routes
import pack_writeback as wb


class DirectorWritebackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'output'
        self.target = self.root / '项目' / '第11集' / '项目_第11集_beat003.zip'
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'original')
        self.service = DirectorWriteback(self.root)
        self.source = self.base / 'MiniMaxH3Director-r2v-xxx.mmxpack.zip'
        self.pack()

    def pack(self, format='minimax-h3-director-pack'):
        with zipfile.ZipFile(self.source, 'w') as archive:
            archive.writestr('pack.json', json.dumps({'format': format}))
            archive.writestr('timeline.json', '{}')

    def ticket(self):
        return self.service.remember(self.target.name)

    def test_import_export_and_backup(self):
        result = self.service.write(self.source, self.ticket())
        self.assertEqual(Path(result['path']), self.target)
        self.assertEqual(self.target.read_bytes(), self.source.read_bytes())
        self.assertEqual(Path(result['backup']).read_bytes(), b'original')
        self.assertRegex(Path(result['backup']).name, r'\.zip\.bak-\d{8}-\d{6}$')

    def test_conventional_export_name(self):
        renamed = self.source.with_name(self.target.name)
        self.source.rename(renamed)
        self.service.write(renamed, self.ticket())
        self.assertEqual(self.target.read_bytes(), renamed.read_bytes())

    def test_missing_identity(self):
        for ticket in (None, {}, {'identity': {}}):
            with self.subTest(ticket=ticket), self.assertRaises(ValueError):
                self.service.write(self.source, ticket)
        self.assertEqual(self.target.read_bytes(), b'original')
        self.assertEqual(list(self.root.rglob('*.bak-*')), [])

    def test_invalid_and_missing_import_names(self):
        for name in ('MiniMaxH3Director-r2v-xxx.mmxpack.zip', '项目_第11集_beat004.zip', '../' + self.target.name):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.service.remember(name)

    def test_ambiguous_import_and_export(self):
        ticket = self.ticket()
        duplicate = self.root / 'other' / self.target.name
        duplicate.parent.mkdir()
        duplicate.write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, '多个同名'):
            self.ticket()
        with self.assertRaisesRegex(ValueError, '多个同名'):
            self.service.write(self.source, ticket)
        self.assertEqual(self.target.read_bytes(), b'original')
        self.assertEqual(duplicate.read_bytes(), b'other')

    def test_wrong_beat_and_escape(self):
        for path in (self.target.with_name('项目_第11集_beat004.zip'), self.base / self.target.name):
            ticket = self.ticket()
            ticket['identity']['path'] = str(path)
            with self.assertRaises(ValueError):
                self.service.write(self.source, ticket)
            # Also exercise the path/identity boundary independently of signature protection.
            ticket['signature'] = self.service._signature(ticket['identity'])
            with self.assertRaises(ValueError):
                self.service.write(self.source, ticket)
        self.assertEqual(self.target.read_bytes(), b'original')

    def test_latest_import(self):
        other = self.target.with_name('项目_第11集_beat004.zip')
        other.write_bytes(b'four')
        self.ticket()
        latest = self.service.remember(other.name)
        self.service.write(self.source, latest)
        self.assertEqual(self.target.read_bytes(), b'original')
        self.assertEqual(other.read_bytes(), self.source.read_bytes())

    def test_invalid_pack_preserves_original(self):
        ticket = self.ticket()
        for kind in ('broken', 'other', 'missing', 'json'):
            if kind == 'broken':
                self.source.write_bytes(b'broken')
            elif kind == 'other':
                self.pack('other')
            else:
                with zipfile.ZipFile(self.source, 'w') as archive:
                    archive.writestr('pack.json', 'invalid' if kind == 'json' else '{}')
                    if kind == 'json':
                        archive.writestr('timeline.json', '{}')
            with self.subTest(kind=kind), self.assertRaises((ValueError, zipfile.BadZipFile)):
                self.service.write(self.source, ticket)
            self.assertEqual(self.target.read_bytes(), b'original')
            self.assertFalse(list(self.root.rglob('*.bak-*')))

    def test_replace_failure_restores_original(self):
        with patch.object(wb.os, 'replace', side_effect=OSError('locked')):
            with self.assertRaises(OSError):
                self.service.write(self.source, self.ticket())
        self.assertEqual(self.target.read_bytes(), b'original')
        self.assertFalse(list(self.root.rglob('.pack-writeback-*')))

    def test_backup_collision_preserves_both(self):
        ticket = self.ticket()
        with patch.object(wb, 'datetime') as clock:
            clock.now.return_value.strftime.return_value = '.bak-20261116-000000'
            backup = self.target.with_name(self.target.name + '.bak-20261116-000000')
            backup.write_bytes(b'previous backup')
            with self.assertRaises(FileExistsError):
                self.service.write(self.source, ticket)
        self.assertEqual(backup.read_bytes(), b'previous backup')
        self.assertEqual(self.target.read_bytes(), b'original')

    def test_register_without_comfy(self):
        with patch.dict('sys.modules', {'server': None}):
            self.assertFalse(register_routes())


if __name__ == '__main__':
    unittest.main()
