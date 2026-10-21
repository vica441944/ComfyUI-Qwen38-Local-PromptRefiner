import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PLANNER = "Qwen38LongVideoDirectorPlanner";
const DIRECTOR_TYPES = new Set(["MiniMaxH3Director", "ComfyMiniMaxH3Director"]);
const CLEAR_GROUP_TYPES = new Set([
  "MiniMaxH3DirectorGroupImageToVideo",
  "MiniMaxH3DirectorGroupReferenceToVideo",
  "MiniMaxH3DirectorGroupsCombine",
]);
const DEFAULT_DIRECTOR_EXECUTION = "Preview plan only (Director blocked)";
const BEAT_PACK_FORMAT = "nvstudio-beat-pack";

function graphOf() {
  return app.graph ?? app.canvas?.graph ?? null;
}

function nodeClass(node) {
  return String(node?.comfyClass ?? node?.type ?? "");
}

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  const element = widget.inputEl ?? widget.element;
  if (element && "value" in element) element.value = value;
  widget.callback?.(value);
}

function setNamedWidget(node, name, value) {
  const widget = (node?.widgets ?? []).find((w) => w?.name === name);
  if (!widget) return false;
  setWidgetValue(widget, value);
  return true;
}

function inputIndex(node, name) {
  return (node?.inputs ?? []).findIndex((input) => input?.name === name);
}

function disconnectInput(node, name) {
  const index = inputIndex(node, name);
  if (index >= 0 && node?.inputs?.[index]?.link != null) node.disconnectInput?.(index);
}

function linkInfo(graph, linkId) {
  if (!graph || linkId == null) return null;
  const links = graph.links;
  let link = links?.[linkId];
  if (!link && typeof links?.find === "function") {
    link = links.find((c) => c && (c.id === linkId || c[0] === linkId));
  }
  if (!link) return null;
  return {
    originId: link.origin_id ?? link[1],
    targetId: link.target_id ?? link[3],
  };
}

function inputSource(graph, node, inputName) {
  const index = inputIndex(node, inputName);
  const input = index >= 0 ? node.inputs[index] : null;
  const link = linkInfo(graph, input?.link);
  if (!link) return null;
  return graph.getNodeById?.(link.originId) ?? null;
}

function findDirector(graph, planner) {
  const outIdx = (planner?.outputs ?? []).findIndex((o) => o?.name === "director_groups");
  const output = outIdx >= 0 ? planner.outputs[outIdx] : null;
  for (const linkId of output?.links ?? []) {
    const link = linkInfo(graph, linkId);
    const target = graph.getNodeById?.(link?.targetId);
    if (DIRECTOR_TYPES.has(nodeClass(target))) return target;
  }
  return (graph?._nodes ?? graph?.nodes ?? []).find((n) => DIRECTOR_TYPES.has(nodeClass(n))) ?? null;
}

function parseManifestClient(data) {
  if (!data || typeof data !== "object") throw new Error("manifest 无效");
  if (data.format !== BEAT_PACK_FORMAT) throw new Error('format 必须是 "nvstudio-beat-pack"');
  if (data.version !== 1) throw new Error("仅支持 beat pack version 1");
  if (typeof data.brief !== "string") throw new Error("brief 必须是字符串");
  const total = Number(data.external_total_duration_seconds);
  const segment = Number(data.segment_duration_seconds);
  if (!Number.isFinite(total) || total < 4) throw new Error("external_total_duration_seconds 无效");
  if (!Number.isFinite(segment) || segment < 4 || segment > 15) {
    throw new Error("segment_duration_seconds 须在 4–15");
  }
  const refs = Array.isArray(data.refs) ? data.refs : [];
  return {
    ...data,
    director_execution: data.director_execution || DEFAULT_DIRECTOR_EXECUTION,
    clear_director_groups: data.clear_director_groups !== false,
    allow_duration_overrun: Boolean(data.allow_duration_overrun),
    refs,
  };
}

function pickBeatPackFile() {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".zip,.json,application/zip,application/json";
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

function readClearGroupsToggle(planner, manifestDefault) {
  // Manifest false always wins. Otherwise the non-serialized UI toggle (default ON).
  if (manifestDefault === false) return false;
  const widget = (planner?.widgets ?? []).find((w) => w?.name === "qwen38_clear_director_groups_on_import");
  if (widget && typeof widget.value === "boolean") return widget.value;
  return true;
}

function clearDirectorGroups(graph, planner) {
  const nodes = [...(graph?._nodes ?? graph?.nodes ?? [])];
  const toRemove = nodes.filter((n) => CLEAR_GROUP_TYPES.has(nodeClass(n)));
  const director = findDirector(graph, planner);
  if (director) {
    disconnectInput(director, "i2v_groups");
    disconnectInput(director, "r2v_groups");
    const editor = director._minimaxEditor;
    if (editor?.timeline) {
      editor.timeline.segments = [];
      editor.timeline.shots = [];
      editor.commit?.(false, { syncTimeline: true });
      editor.updateSelectionUI?.();
      editor.scheduleRender?.();
    } else {
      const tw = (director.widgets ?? []).find((w) => w?.name === "timeline_data");
      if (tw && typeof tw.value === "string") {
        try {
          const data = JSON.parse(tw.value || "{}");
          data.segments = [];
          data.shots = [];
          setWidgetValue(tw, JSON.stringify(data));
        } catch (_) {
          /* keep existing timeline_data if unparsable */
        }
      }
    }
  }
  for (const node of toRemove) graph.remove?.(node);
  return toRemove.length;
}

const REFINER_NODE_TYPES = new Set([
  "Qwen38LocalPromptRefiner",
  "Qwen38DirectorGroupReRefiner",
]);

/**
 * After "生成原生 Director Groups", Planner (and Group Re-Refiner after one run)
 * are set to LiteGraph NEVER so a Director Queue skips another Qwen inference.
 * Beat-pack import replaces the brief and clears those Groups — without arming
 * Planner/refiners back to ALWAYS, the next Queue leaves them stuck on "Never"
 * and the new beat never re-plans / re-refines.
 */
function armPlannerAndRefinersForNewBeat(graph, planner, { clearGroups } = {}) {
  const ALWAYS = globalThis.LiteGraph?.ALWAYS ?? 0;
  const NEVER = globalThis.LiteGraph?.NEVER ?? 2;

  if (planner) {
    planner.mode = ALWAYS;
    planner.properties ??= {};
    delete planner.properties.qwen38DirectorMaterialized;
    // Specs belonged to the previous beat (Groups may have been removed).
    delete planner.properties.qwen38DirectorGroupSpecs;
    planner._qwen38DirectorGroupSpecs = undefined;
    planner.setDirtyCanvas?.(true, true);
  }

  for (const node of graph?._nodes ?? graph?.nodes ?? []) {
    const cls = nodeClass(node);
    if (!REFINER_NODE_TYPES.has(cls)) continue;
    if (cls === "Qwen38DirectorGroupReRefiner") {
      if (clearGroups) {
        // Groups were cleared — drop stale target so a leftover Always cannot
        // write into a deleted id. Stay Never until the user reads a new Group
        // (that path already arms Always for one run).
        node.mode = NEVER;
        node._qwen38RevisedPrompt = "";
        node.properties ??= {};
        delete node.properties.qwen38GroupRevisionSource;
        delete node.properties.qwen38LastRevision;
        setNamedWidget(node, "target_group_id", "");
        setNamedWidget(node, "current_prompt", "");
        setNamedWidget(node, "previous_group_context", "");
        setNamedWidget(node, "next_group_context", "");
      }
    } else {
      // Local Prompt Refiner: force Always so it participates in the new beat.
      node.mode = ALWAYS;
    }
    node.setDirtyCanvas?.(true, true);
  }
}

function writeBrief(graph, planner, brief) {
  const linked = inputSource(graph, planner, "external_brief");
  if (linked) {
    if (
      setNamedWidget(linked, "value", brief)
      || setNamedWidget(linked, "text", brief)
      || setNamedWidget(linked, "string", brief)
    ) {
      return "external_brief";
    }
  }
  setNamedWidget(planner, "brief", brief);
  return "brief";
}

function writeDuration(graph, planner, seconds) {
  const linked = inputSource(graph, planner, "external_total_duration_seconds");
  if (linked) {
    if (setNamedWidget(linked, "value", seconds) || setNamedWidget(linked, "number", seconds)) {
      return "external_total_duration_seconds";
    }
  }
  setNamedWidget(planner, "total_duration_seconds", seconds);
  return "total_duration_seconds";
}

function slotTitleAliases(slot) {
  const match = /^reference_(image|video|audio)_(\d+)$/i.exec(String(slot || "").trim());
  if (!match) return [];
  const kind = match[1].toLowerCase();
  const index = match[2];
  if (kind === "image") return [`Ref_i_${index}`, `Ref ${index}`, `Ref_${index}`];
  if (kind === "video") return [`Ref_v_${index}`];
  if (kind === "audio") return [`Ref_a_${index}`];
  return [];
}

function aliasInTitle(title, alias) {
  const rawTitle = String(title ?? "");
  const rawAlias = String(alias ?? "");
  if (!rawTitle || !rawAlias) return false;
  if (rawTitle.toLowerCase() === rawAlias.toLowerCase()) return true;
  let body;
  const spaced = /^Ref (\d+)$/i.exec(rawAlias);
  if (spaced) {
    body = `Ref[\\s_]+${spaced[1]}`;
  } else {
    body = rawAlias.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }
  const re = new RegExp(`(?:^|[^A-Za-z0-9_])${body}(?:[^A-Za-z0-9_]|$)`, "i");
  if (re.test(rawTitle)) return true;
  return rawTitle.toLowerCase().includes(rawAlias.toLowerCase());
}

function titleMatchesSlot(title, slot) {
  const t = String(title ?? "").trim();
  const s = String(slot ?? "").trim();
  if (!t || !s) return false;
  if (t === s) return true;
  const canonical = new RegExp(`(?:^|[^A-Za-z0-9_])${s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?:[^A-Za-z0-9_]|$)`, "i");
  if (canonical.test(t) || t.toLowerCase().includes(s.toLowerCase())) return true;
  for (const alias of slotTitleAliases(s)) {
    if (aliasInTitle(t, alias)) return true;
  }
  return false;
}
function loaderPathWidgetNames(kind, nodeClassName) {
  // Only the path combo for this media kind — never probe audio on LoadImage
  // (Ref_i_*) or image on LoadAudio (Ref_a_*), which caused dual "缺少输入".
  const cls = String(nodeClassName || "");
  if (kind === "image") return ["image", "filename", "file", "path"];
  if (kind === "video") return ["video", "filename", "file", "path"];
  if (kind === "audio") return ["audio", "filename", "file", "path"];
  if (/LoadImage/i.test(cls)) return ["image", "filename", "file", "path"];
  if (/LoadAudio/i.test(cls)) return ["audio", "filename", "file", "path"];
  if (/VHS_LoadVideo|LoadVideo/i.test(cls)) return ["video", "filename", "file", "path"];
  return ["filename", "file", "path"];
}

function ensureComboOption(widget, pathValue) {
  // Programmatic paths (qwen38_beat_packs/…) are missing from the cached combo
  // list until refresh — Comfy then red-boxes the node as "缺少输入".
  if (!widget) return;
  widget.options ??= {};
  const opts = widget.options;
  let values = opts.values;
  if (typeof values === "function") {
    try {
      values = values();
    } catch (_) {
      values = null;
    }
  }
  if (!Array.isArray(values)) {
    values = values != null && values !== "" ? [values] : [];
  }
  if (!values.includes(pathValue)) values = [...values, pathValue];
  opts.values = values;
  // Some Comfy builds keep a parallel list used by the combo UI.
  if (Array.isArray(opts.values) && Array.isArray(widget.options?.values)) {
    widget.options.values = values;
  }
}

function refreshLoaderPathWidget(node, widget, pathValue) {
  if (!widget) return;
  ensureComboOption(widget, pathValue);
  // Re-apply after options update so validation sees a listed value.
  setWidgetValue(widget, pathValue);
  try {
    widget.callback?.(pathValue, undefined, node, undefined, undefined);
  } catch (_) {
    try { widget.callback?.(pathValue); } catch (__) { /* ignore */ }
  }
  // Drop stale missing-input flags if the frontend stored them on the widget.
  delete widget._invalid;
  delete widget.validationError;
  node.setDirtyCanvas?.(true, true);
}

function applyLoaderPath(node, kind, pathValue) {
  const names = loaderPathWidgetNames(kind, nodeClass(node));
  for (const name of names) {
    const widget = (node?.widgets ?? []).find((w) => w?.name === name);
    if (!widget) continue;
    // Only refresh this existing path widget for this kind — do not touch
    // other media combos on the same node (if any).
    refreshLoaderPathWidget(node, widget, pathValue);
    return name;
  }
  return null;
}

function fillLoaders(graph, refs) {
  const nodes = graph?._nodes ?? graph?.nodes ?? [];
  const report = [];
  for (const ref of refs ?? []) {
    const slot = String(ref.slot ?? "");
    const kind = String(ref.kind ?? "");
    const pathValue = String(ref.input_path || ref.path || "").replace(/\\/g, "/");
    if (!slot || !pathValue) {
      report.push({ slot, ok: false, reason: "missing path" });
      continue;
    }
    const match = nodes.find((n) => titleMatchesSlot(n?.title, slot));
    if (!match) {
      report.push({ slot, ok: false, reason: "no Load* node title match" });
      continue;
    }
    const widget = applyLoaderPath(match, kind, pathValue);
    if (!widget) {
      report.push({
        slot,
        ok: false,
        reason: `matched ${match.title} but no path widget`,
        node: match.title,
      });
      continue;
    }
    report.push({ slot, ok: true, node: match.title, widget, path: pathValue });
  }
  graph?.setDirtyCanvas?.(true, true);
  return report;
}

async function loadPackFromFile(file) {
  const name = String(file?.name || "").toLowerCase();
  if (name.endsWith(".json")) {
    const text = await file.text();
    const manifest = parseManifestClient(JSON.parse(text));
    return { manifest, refs: manifest.refs.map((r) => ({ ...r })) };
  }
  const body = new FormData();
  body.append("pack", file, file.name || "beat.nvstudio-beat-pack.zip");
  const resp = await api.fetchApi("/qwen38/promptrefiner/import_beat_pack", { method: "POST", body });
  const data = await resp.json().catch(async () => ({ error: await resp.text() }));
  if (!resp.ok) throw new Error(data?.error || `导入失败 HTTP ${resp.status}`);
  const manifest = parseManifestClient(data.manifest);
  return { manifest, refs: data.refs || manifest.refs };
}

function applyBeatPack(planner, pack, { clearGroups }) {
  const graph = graphOf();
  if (!graph) throw new Error("ComfyUI 画布尚未就绪");
  let removed = 0;
  if (clearGroups) removed = clearDirectorGroups(graph, planner);

  const briefTarget = writeBrief(graph, planner, pack.manifest.brief);
  const durationTarget = writeDuration(
    graph,
    planner,
    Number(pack.manifest.external_total_duration_seconds),
  );
  setNamedWidget(planner, "segment_duration_seconds", Number(pack.manifest.segment_duration_seconds));
  setNamedWidget(planner, "allow_duration_overrun", Boolean(pack.manifest.allow_duration_overrun));
  setNamedWidget(
    planner,
    "director_execution",
    pack.manifest.director_execution || DEFAULT_DIRECTOR_EXECUTION,
  );

  const loaderReport = fillLoaders(graph, pack.refs);
  // New brief + cleared Groups require Planner (and related refiners) to leave
  // Never and re-run on the next Queue — same intent as manually switching
  // the Qwen node back to Always after materialization.
  armPlannerAndRefinersForNewBeat(graph, planner, { clearGroups });
  graph.setDirtyCanvas?.(true, true);
  planner.setDirtyCanvas?.(true, true);
  return { removed, briefTarget, durationTarget, loaderReport };
}

async function importBeatPackForPlanner(planner) {
  const file = await pickBeatPackFile();
  if (!file) return;
  const pack = await loadPackFromFile(file);
  const clearGroups = readClearGroupsToggle(planner, pack.manifest.clear_director_groups !== false);
  const result = applyBeatPack(planner, pack, { clearGroups });
  const okLoads = result.loaderReport.filter((r) => r.ok).length;
  const miss = result.loaderReport.filter((r) => !r.ok);
  const missText = miss.length
    ? `\n未匹配 Load*：\n${miss.map((m) => `- ${m.slot}: ${m.reason}`).join("\n")}`
    : "";
  globalThis.alert?.(
    `节拍包已导入（未替换工作流、未自动 Queue）\n\n`
      + `clear_director_groups: ${clearGroups ? "是" : "否"}`
      + (clearGroups ? `（删除 Group/Combine ${result.removed} 个）` : "")
      + `\nbrief → ${result.briefTarget}\n`
      + `时长 → ${result.durationTarget}\n`
      + `Load* 已填 ${okLoads}/${result.loaderReport.length}`
      + `\nPlanner 已切回 Always（可 Queue 重跑新节拍）`
      + missText,
  );
  globalThis.Qwen38PromptRefiner?.notifyRefsChanged?.(planner);
}

function addBeatPackControls(planner) {
  if (planner?._qwen38BeatPackControlsAdded) return;
  planner._qwen38BeatPackControlsAdded = true;
  // Non-serialized: default ON, user can uncheck. Does not shift backend widget order.
  const clearToggle = planner.addWidget?.(
    "toggle",
    "导入时清除 Director Groups",
    true,
    () => {},
    { serialize: false },
  );
  if (clearToggle) clearToggle.name = "qwen38_clear_director_groups_on_import";
  const run = () => {
    importBeatPackForPlanner(planner).catch((error) => {
      console.error("[Qwen38] beat pack import failed", error);
      globalThis.alert?.(`导入节拍包失败：${error?.message ?? error}`);
    });
  };
  planner.addWidget?.("button", "导入节拍包", null, run, { serialize: false });
  const originalMenu = planner.getExtraMenuOptions;
  planner.getExtraMenuOptions = function (...args) {
    const options = originalMenu?.apply(this, args) ?? [];
    options.push({ content: "Import beat pack / 导入节拍包", callback: run });
    return options;
  };
}


// Shared helpers for refs_panel.js (same WEB_DIRECTORY; no bundler).
globalThis.Qwen38PromptRefiner = Object.assign(globalThis.Qwen38PromptRefiner || {}, {
  graphOf,
  nodeClass,
  titleMatchesSlot,
  slotTitleAliases,
  loaderPathWidgetNames,
  applyLoaderPath,
  refreshLoaderPathWidget,
  ensureComboOption,
  setWidgetValue,
  PLANNER: "Qwen38LongVideoDirectorPlanner",
  notifyRefsChanged(planner) {
    try {
      globalThis.Qwen38PromptRefiner?._refsPanelRefresh?.(planner);
    } catch (error) {
      console.warn("[Qwen38] refs panel refresh failed", error);
    }
  },
});

app.registerExtension({
  name: "Qwen38LocalPromptRefiner.BeatPackImport",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== PLANNER) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      addBeatPackControls(this);
      return result;
    };
  },
});
