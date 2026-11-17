import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const code = fs.readFileSync(new URL('../web/director_pack_writeback.js', import.meta.url), 'utf8').replace(/^import .*;\r?\n/gm, '');
const calls = [];
const notices = [];
let change;
let failWrite = false;
const api = {
  async fetchApi(route, options) {
    calls.push({ route, options });
    if (route.endsWith('/import_pack')) return Response.json({ timeline: {} });
    if (route.endsWith('/upload_chunk')) return Response.json({ name: 'uploaded.zip' });
    if (route.endsWith('/director_import_identity')) {
      const name = JSON.parse(options.body).name;
      return name.includes('_beat') ? Response.json({ identity: { path: name }, signature: 'test' })
        : Response.json({ error: 'invalid identity' }, { status: 400 });
    }
    if (route.endsWith('/director_export_writeback')) return failWrite
      ? Response.json({ error: 'write failed' }, { status: 400 })
      : Response.json({ path: 'test-output', backup: 'test-backup' });
    if (route.endsWith('/export_pack')) {
      const data = JSON.parse(options.body);
      if (data.dryRun || data.dry_run) return Response.json({ totalBytes: 3 });
      if (data.binary) return new Response('zip', { headers: { 'Content-Type': 'application/zip' } });
      return Response.json({ zipB64: btoa('zip'), downloadName: 'MiniMaxH3Director-r2v-test.mmxpack.zip' });
    }
    throw new Error('Unexpected route');
  },
};
vm.runInNewContext(code, {
  api, app: { registerExtension(extension) { extension.setup(); } },
  FormData, Blob, Uint8Array, atob,
  alert: text => notices.push(text),
  document: { addEventListener(type, handler) { change = handler; } },
});
const exported = (data = {}) => api.fetchApi('/minimax/director/export_pack', { body: JSON.stringify(data) });
const writes = () => calls.filter(c => c.route.endsWith('/director_export_writeback'));
const imported = async name => {
  const body = new FormData();
  body.append('pack', new Blob(['zip']), name);
  return api.fetchApi('/minimax/director/import_pack', { body });
};
assert.equal((await (await exported()).json()).zipB64, btoa('zip'));
assert.equal(writes().length, 0);
await imported('项目_第11集_beat003.zip');
await exported({ dryRun: true });
await exported({ dry_run: true });
assert.equal(writes().length, 0);
assert.equal((await (await exported()).json()).zipB64, btoa('zip'));
assert.equal(writes().length, 1);
assert.equal(JSON.parse(writes().at(-1).options.body.get('identity')).identity.path, '项目_第11集_beat003.zip');
await imported('项目_第11集_beat004.zip');
assert.equal(await (await exported({ binary: true })).text(), 'zip');
assert.equal(JSON.parse(writes().at(-1).options.body.get('identity')).identity.path, '项目_第11集_beat004.zip');
failWrite = true;
assert.equal((await (await exported()).json()).zipB64, btoa('zip'));
assert.ok(notices.at(-1).includes('已保留 Director 默认下载'));
failWrite = false;
await imported('unknown.zip');
const before = writes().length;
await exported();
assert.equal(writes().length, before);
change({ target: { type: 'file', files: [{ name: '项目_第11集_beat007.zip' }] } });
const chunk = new FormData();
chunk.set('upload_id', 'test');
chunk.set('chunk_index', '0');
chunk.set('filename', '项目_第11集_beat007.zip');
await api.fetchApi('/minimax/director/upload_chunk', { body: chunk });
await api.fetchApi('/minimax/director/import_pack', { body: JSON.stringify({ filename: 'uploaded.zip' }) });
await exported();
assert.equal(JSON.parse(writes().at(-1).options.body.get('identity')).identity.path, '项目_第11集_beat007.zip');
console.log('Frontend hook checks passed: direct/chunked import, latest/invalid identity, dry runs, JSON/binary exports, download fallback.');
