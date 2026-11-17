import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "Qwen38LocalPromptRefiner.DirectorWriteback",
  setup() {
    const fetchApi = api.fetchApi;
    let identity = null;
    let generation = 0;
    const uploads = new Map();
    const chunkNames = new Map();
    let selectedName = null;
    document.addEventListener("change", (event) => {
      if (event.target?.type === "file") selectedName = event.target.files?.[0]?.name || null;
    }, true);
    const prefix = "/qwen38/promptrefiner/";
    const notice = (text) => globalThis.alert?.(text);
    api.fetchApi = async function (route, options = {}) {
      const isImport = route === "/minimax/director/import_pack";
      const isExport = route === "/minimax/director/export_pack";
      const isChunk = route === "/minimax/director/upload_chunk";
      if (!isImport && !isExport && !isChunk) return fetchApi.call(this, route, options);
      const body = options.body;
      const form = body instanceof FormData;
      let data = {};
      if (!form && typeof body === "string") {
        try { data = JSON.parse(body); } catch { /* Director handles malformed requests. */ }
      }
      const importedName = isImport ? (form ? body.get("pack")?.name : uploads.get(data.filename)) : null;
      if (isChunk && form && body.get("chunk_index") === "0") {
        // Director truncates upload stems to 80 characters; retain the actual file.name.
        const uploadName = selectedName?.endsWith(".zip") ? selectedName.slice(0, -4).slice(0, 80) + ".zip" : null;
        chunkNames.set(body.get("upload_id"), uploadName === body.get("filename") ? selectedName : null);
        if (chunkNames.size > 16) chunkNames.delete(chunkNames.keys().next().value);
      }
      const exportIdentity = identity;
      const exportGeneration = generation;
      const response = await fetchApi.call(this, route, options);
      if (!response.ok) return response;
      try {
        if (isChunk && form) {
          const result = await response.clone().json();
          if (result.name) {
            uploads.set(result.name, chunkNames.get(body.get("upload_id")));
            chunkNames.delete(body.get("upload_id"));
            if (uploads.size > 16) uploads.delete(uploads.keys().next().value);
          }
        }
        if (isImport) {
          const result = await response.clone().json();
          if (!result?.timeline) return response;
          const current = ++generation;
          identity = null;
          const remembered = await fetchApi.call(this, prefix + "director_import_identity", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: importedName || "" }),
          });
          const ticket = await remembered.json();
          if (current === generation && remembered.ok) identity = ticket;
          if (!remembered.ok) notice(`${ticket.error}\n自动写回未启用，Director 默认下载仍可用。`);
        }
        if (isExport && !data.dryRun && !data.dry_run) {
          if (!exportIdentity || exportGeneration !== generation) throw new Error("无有效导入身份或导入已改变，请从 output 用约定文件名再导入一次");
          const copy = response.clone();
          let blob;
          if ((copy.headers.get("Content-Type") || "").includes("application/json")) {
            const result = await copy.json();
            if (!result.zipB64) throw new Error("导出结果缺少 ZIP");
            blob = new Blob([Uint8Array.from(atob(result.zipB64), c => c.charCodeAt(0))], { type: "application/zip" });
          } else {
            blob = await copy.blob();
          }
          if (exportGeneration !== generation) throw new Error("导入已改变，拒绝写回");
          const upload = new FormData();
          upload.append("identity", JSON.stringify(exportIdentity));
          upload.append("pack", blob, "Director-export.zip");
          const written = await fetchApi.call(this, prefix + "director_export_writeback", { method: "POST", body: upload });
          const result = await written.json();
          if (!written.ok) throw new Error(result.error || "写回失败");
          notice(`已写回：${result.path}\n备份：${result.backup || "无"}\nDirector 默认下载仍会保留，无需手动改名或复制。`);
        }
      } catch (error) {
        if (isImport) { identity = null; generation++; }
        notice(`自动写回未完成：${error?.message || "未知错误"}\n已保留 Director 默认下载。`);
      }
      // Leave the original response untouched so Director always retains its download.
      return response;
    };
  },
});
