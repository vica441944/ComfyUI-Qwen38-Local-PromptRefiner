import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Lightweight "本拍 refs" panel on Qwen38LongVideoDirectorPlanner.
 * Lists existing Load* nodes matched by title/aliases; preview + replace/fill path.
 * NEVER creates Load* nodes. Empty slot = matched Load* with empty path.
 */

const PLANNER = "Qwen38LongVideoDirectorPlanner";
const PANEL_PROP = "qwen38RefsPanelOpen";

function api_() {
  return globalThis.Qwen38PromptRefiner || {};
}

function graphOf() {
  return api_().graphOf?.() ?? app.graph ?? app.canvas?.graph ?? null;
}

function nodeClass(node) {
  return api_().nodeClass?.(node) ?? String(node?.comfyClass ?? node?.type ?? "");
}

function titleMatchesSlot(title, slot) {
  if (api_().titleMatchesSlot) return api_().titleMatchesSlot(title, slot);
  return false;
}

function applyLoaderPath(node, kind, pathValue) {
  if (api_().applyLoaderPath) return api_().applyLoaderPath(node, kind, pathValue);
  return null;
}

/** Parse canonical slot from a Load* node title. Prefer typed aliases over bare Ref N. */
function slotFromTitle(title) {
  const t = String(title ?? "").trim();
  if (!t) return null;
  const canonical = t.match(/reference_(image|video|audio)_(\d+)/i);
  if (canonical) {
    return {
      slot: `reference_${canonical[1].toLowerCase()}_${canonical[2]}`,
      kind: canonical[1].toLowerCase(),
      index: Number(canonical[2]),
    };
  }
  const typed = t.match(/(?:^|[^A-Za-z0-9_])Ref_([iva])_(\d+)(?:[^A-Za-z0-9_]|$)/i);
  if (typed) {
    const kind = { i: "image", v: "video", a: "audio" }[typed[1].toLowerCase()];
    return { slot: `reference_${kind}_${typed[2]}`, kind, index: Number(typed[2]) };
  }
  // Bare "Ref N" / "Ref_N" → image only (same rule as beat-pack aliases).
  const bare = t.match(/(?:^|[^A-Za-z0-9_])Ref[\s_]+(\d+)(?:[^A-Za-z0-9_]|$)/i);
  if (bare) {
    return { slot: `reference_image_${bare[1]}`, kind: "image", index: Number(bare[1]) };
  }
  return null;
}

function pathWidgetNames(kind, cls) {
  if (api_().loaderPathWidgetNames) return api_().loaderPathWidgetNames(kind, cls);
  if (kind === "image") return ["image", "filename", "file", "path"];
  if (kind === "video") return ["video", "filename", "file", "path"];
  if (kind === "audio") return ["audio", "filename", "file", "path"];
  return ["filename", "file", "path"];
}

function readPath(node, kind) {
  const names = pathWidgetNames(kind, nodeClass(node));
  for (const name of names) {
    const w = (node?.widgets ?? []).find((x) => x?.name === name);
    if (w && w.value != null && String(w.value).trim() !== "") {
      return { widgetName: name, path: String(w.value).replace(/\\/g, "/") };
    }
  }
  // Empty path: still report first existing path widget name for fill.
  for (const name of names) {
    const w = (node?.widgets ?? []).find((x) => x?.name === name);
    if (w) return { widgetName: name, path: "" };
  }
  return { widgetName: null, path: "" };
}

/**
 * Scan graph for Load* whose titles match reference_* / Ref_i|v|a_* / Ref N.
 * Does NOT invent nodes — only lists what already exists.
 */
function scanBeatRefs(graph) {
  const nodes = graph?._nodes ?? graph?.nodes ?? [];
  const bySlot = new Map();
  for (const node of nodes) {
    const parsed = slotFromTitle(node?.title);
    if (!parsed) continue;
    // Prefer nodes that are clearly loaders; still allow any title-matched node
    // that has an image/video/audio path widget (user may rename class titles).
    const pathInfo = readPath(node, parsed.kind);
    if (!pathInfo.widgetName) continue;
    const prev = bySlot.get(parsed.slot);
    // Prefer a filled path over an empty one if duplicates.
    if (prev && prev.path && !pathInfo.path) continue;
    bySlot.set(parsed.slot, {
      slot: parsed.slot,
      kind: parsed.kind,
      index: parsed.index,
      nodeId: node.id,
      title: String(node.title ?? ""),
      path: pathInfo.path,
      widgetName: pathInfo.widgetName,
      empty: !pathInfo.path,
    });
  }
  return [...bySlot.values()].sort((a, b) => {
    const order = { image: 0, video: 1, audio: 2 };
    return (order[a.kind] - order[b.kind]) || (a.index - b.index);
  });
}

function parseComfyMediaPath(pathValue) {
  const raw = String(pathValue || "").replace(/\\/g, "/").trim();
  if (!raw) return null;
  const parts = raw.split("/");
  const filename = parts.pop();
  const subfolder = parts.join("/");
  return { filename, subfolder, type: "input" };
}

function mediaViewUrl(kind, pathValue) {
  const parsed = parseComfyMediaPath(pathValue);
  if (!parsed?.filename) return null;
  if (kind !== "image") return null;
  const q = new URLSearchParams({
    filename: parsed.filename,
    subfolder: parsed.subfolder || "",
    type: parsed.type || "input",
  });
  return api.apiURL(`/view?${q}`);
}

function acceptForKind(kind) {
  if (kind === "image") return "image/*";
  if (kind === "video") return "video/*,.mp4,.webm,.mov,.mkv";
  if (kind === "audio") return "audio/*,.wav,.mp3,.flac,.ogg,.m4a";
  return "*/*";
}

function pickLocalFile(accept) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept || "*/*";
    input.style.display = "none";
    document.body.appendChild(input);
    input.addEventListener("change", () => {
      const file = input.files?.[0] || null;
      input.remove();
      resolve(file);
    }, { once: true });
    input.click();
  });
}

/** Upload into input/qwen38_beat_packs/refs_panel/ via Comfy /upload/image (works for binary too). */
async function uploadRefFile(file, kind) {
  const body = new FormData();
  const safeName = String(file.name || `ref_${kind}`).replace(/[^\w.\-()+]+/g, "_");
  body.append("image", file, safeName);
  body.append("overwrite", "true");
  body.append("type", "input");
  body.append("subfolder", "qwen38_beat_packs/refs_panel");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `upload failed HTTP ${resp.status}`);
  }
  const data = await resp.json();
  const name = data.name || data.filename || safeName;
  const sub = data.subfolder || "qwen38_beat_packs/refs_panel";
  const pathValue = sub ? `${String(sub).replace(/\\/g, "/")}/${name}` : name;
  return pathValue.replace(/\\/g, "/");
}

function findNode(graph, nodeId) {
  return graph?.getNodeById?.(nodeId)
    ?? (graph?._nodes ?? graph?.nodes ?? []).find((n) => n?.id === nodeId)
    ?? null;
}

async function replaceOrFillRef(entry) {
  const graph = graphOf();
  const node = findNode(graph, entry.nodeId);
  if (!node) throw new Error(`图上找不到节点 #${entry.nodeId}（${entry.slot}）`);
  // Hard rule: never create Load* — only write back to this existing node.
  const file = await pickLocalFile(acceptForKind(entry.kind));
  if (!file) return null;
  const pathValue = await uploadRefFile(file, entry.kind);
  const widgetName = applyLoaderPath(node, entry.kind, pathValue);
  if (!widgetName) throw new Error(`节点 ${entry.title} 没有可写的 ${entry.kind} 路径控件`);
  try {
    node.onWidgetChanged?.(widgetName, pathValue, entry.path || "", (node.widgets ?? []).find((w) => w?.name === widgetName));
  } catch (_) { /* older frontends */ }
  graph?.setDirtyCanvas?.(true, true);
  return pathValue;
}

function el(tag, style, text) {
  const node = document.createElement(tag);
  if (style) Object.assign(node.style, style);
  if (text != null) node.textContent = text;
  return node;
}

function buildPanelDom(planner) {
  const root = el("div", {
    padding: "8px",
    marginTop: "6px",
    border: "1px solid #444",
    borderRadius: "8px",
    background: "#1e1e1e",
    color: "#ddd",
    fontSize: "12px",
    maxWidth: "420px",
    boxSizing: "border-box",
  });
  const head = el("div", {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "8px",
    marginBottom: "6px",
  });
  head.appendChild(el("strong", { fontSize: "13px" }, "本拍 refs"));
  const hint = el("span", { opacity: "0.7", fontSize: "11px" }, "仅绑已有 Load* · 不自动新建");
  head.appendChild(hint);
  const refreshBtn = el("button", {
    cursor: "pointer",
    padding: "2px 8px",
  }, "刷新");
  refreshBtn.type = "button";
  head.appendChild(refreshBtn);
  root.appendChild(head);

  const list = el("div", { display: "flex", flexDirection: "column", gap: "6px" });
  root.appendChild(list);

  const emptyNote = el("div", { opacity: "0.65", padding: "4px 0" }, "");
  root.appendChild(emptyNote);

  async function render() {
    list.innerHTML = "";
    const refs = scanBeatRefs(graphOf());
    if (!refs.length) {
      emptyNote.textContent = "图上没有 title 匹配的 Load*（reference_* / Ref_i_* / Ref_v_* / Ref_a_* / Ref N）。请先放好 Load* 并改标题。";
      return;
    }
    emptyNote.textContent = `共 ${refs.length} 个槽（空路径可「填」；有路径可「换」）。`;
    for (const entry of refs) {
      const row = el("div", {
        display: "grid",
        gridTemplateColumns: "56px 1fr auto",
        gap: "8px",
        alignItems: "center",
        padding: "6px",
        background: "#2a2a2a",
        borderRadius: "6px",
        border: entry.empty ? "1px dashed #888" : "1px solid #333",
      });

      const thumbWrap = el("div", {
        width: "56px",
        height: "56px",
        background: "#111",
        borderRadius: "4px",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: "11px",
        color: "#aaa",
        textAlign: "center",
      });
      if (entry.kind === "image" && entry.path) {
        const img = document.createElement("img");
        img.src = mediaViewUrl("image", entry.path) || "";
        img.alt = entry.slot;
        Object.assign(img.style, { width: "100%", height: "100%", objectFit: "cover" });
        img.onerror = () => {
          thumbWrap.textContent = "IMG";
        };
        thumbWrap.appendChild(img);
      } else if (entry.kind === "video") {
        thumbWrap.textContent = entry.path ? "VIDEO" : "空视频";
      } else if (entry.kind === "audio") {
        thumbWrap.textContent = entry.path ? "AUDIO" : "空音频";
      } else {
        thumbWrap.textContent = entry.empty ? "空" : "·";
      }
      row.appendChild(thumbWrap);

      const meta = el("div", { minWidth: "0" });
      meta.appendChild(el("div", { fontWeight: "600" }, `${entry.slot}${entry.empty ? "（空）" : ""}`));
      meta.appendChild(el("div", {
        opacity: "0.75",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
      }, entry.title));
      meta.appendChild(el("div", {
        opacity: "0.65",
        fontSize: "11px",
        whiteSpace: "nowrap",
        overflow: "hidden",
        textOverflow: "ellipsis",
      }, entry.path || "（路径为空 — 可在面板填入）"));
      row.appendChild(meta);

      const action = el("button", {
        cursor: "pointer",
        padding: "4px 8px",
        whiteSpace: "nowrap",
      }, entry.empty ? "填文件" : (entry.kind === "image" ? "换图" : "换文件"));
      action.type = "button";
      action.addEventListener("click", async () => {
        action.disabled = true;
        try {
          const path = await replaceOrFillRef(entry);
          if (path) render();
        } catch (error) {
          console.error("[Qwen38] refs panel replace failed", error);
          globalThis.alert?.(`更换失败：${error?.message ?? error}`);
        } finally {
          action.disabled = false;
        }
      });
      row.appendChild(action);
      list.appendChild(row);
    }
  }

  refreshBtn.addEventListener("click", () => render());
  root._qwen38RenderRefs = render;
  render();
  return root;
}

function mountRefsPanel(planner) {
  if (planner?._qwen38RefsPanelMounted) return;
  planner._qwen38RefsPanelMounted = true;
  planner.properties ??= {};

  const toggle = () => {
    const open = !planner.properties[PANEL_PROP];
    planner.properties[PANEL_PROP] = open;
    if (open) showDom(planner);
    else hideDom(planner);
  };

  planner.addWidget?.("button", "本拍 refs", null, toggle, { serialize: false });
  const originalMenu = planner.getExtraMenuOptions;
  planner.getExtraMenuOptions = function (...args) {
    const options = originalMenu?.apply(this, args) ?? [];
    options.push({ content: "本拍 refs 面板", callback: toggle });
    return options;
  };

  // Auto-open once if previously open in properties (optional; default closed).
  if (planner.properties[PANEL_PROP]) {
    queueMicrotask(() => showDom(planner));
  }
}

function ensurePanel(planner) {
  if (planner._qwen38RefsPanelEl?._qwen38RenderRefs) return planner._qwen38RefsPanelEl;
  const panel = buildPanelDom(planner);
  panel.dataset.qwen38RefsPanel = "1";
  if (typeof planner.addDOMWidget === "function") {
    const domWidget = planner.addDOMWidget("qwen38_refs_panel", "div", panel, { serialize: false });
    planner._qwen38RefsDomWidget = domWidget;
    planner._qwen38RefsPanelEl = panel;
    planner.setSize?.(planner.computeSize?.() ?? planner.size);
  } else {
    const canvas = app.canvas?.canvas ?? document.querySelector("canvas");
    const parent = canvas?.parentElement ?? document.body;
    Object.assign(panel.style, {
      position: "absolute",
      zIndex: "40",
      left: `${(planner.pos?.[0] ?? 0) + (planner.size?.[0] ?? 300) + 12}px`,
      top: `${planner.pos?.[1] ?? 40}px`,
    });
    parent.appendChild(panel);
    planner._qwen38RefsPanelEl = panel;
  }
  return panel;
}

function showDom(planner) {
  const panel = ensurePanel(planner);
  panel.style.display = "";
  panel._qwen38RenderRefs?.();
  planner.setDirtyCanvas?.(true, true);
}

function hideDom(planner) {
  const panel = planner?._qwen38RefsPanelEl;
  if (panel) panel.style.display = "none";
  planner.setDirtyCanvas?.(true, true);
}

function refreshPanel(planner) {
  const target = planner
    ?? (graphOf()?._nodes ?? []).find((n) => (n?.comfyClass ?? n?.type) === PLANNER);
  if (!target) return;
  if (target._qwen38RefsPanelEl?._qwen38RenderRefs) {
    target._qwen38RefsPanelEl._qwen38RenderRefs();
    return;
  }
  // Panel closed: still fine; next open rescans. Optionally auto-open after import.
  if (target.properties?.[PANEL_PROP]) showDom(target);
}

globalThis.Qwen38PromptRefiner = Object.assign(globalThis.Qwen38PromptRefiner || {}, {
  _refsPanelRefresh: refreshPanel,
});

app.registerExtension({
  name: "Qwen38LocalPromptRefiner.RefsPanel",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== PLANNER) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      // Wait a tick so BeatPackImport can install helpers first when both load.
      queueMicrotask(() => mountRefsPanel(this));
      return result;
    };
  },
  async setup() {
    const attach = () => {
      for (const n of graphOf()?._nodes ?? []) {
        if ((n?.comfyClass ?? n?.type) === PLANNER) mountRefsPanel(n);
      }
    };
    queueMicrotask(attach);
    setTimeout(attach, 500);
  },
});
