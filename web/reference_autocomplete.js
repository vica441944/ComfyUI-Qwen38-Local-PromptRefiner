import { app } from "../../scripts/app.js";

const NODE_NAMES = new Set(["Qwen38LocalPromptRefiner", "Qwen38LongVideoDirectorPlanner"]);
const WIDGET_NAME = "brief";
const EXTERNAL_BRIEF_INPUT = "external_brief";
const ASSET_TYPES = [
  { prefix: "reference_image_", icon: "🖼", label: "Picture" },
  { prefix: "reference_video_", icon: "🎞", label: "Video" },
  { prefix: "reference_video_frames_", tokenPrefix: "reference_video_", icon: "🎞", label: "Director video" },
  { prefix: "reference_video_audio_", icon: "🔉", label: "Video audio" },
  { prefix: "reference_audio_", icon: "🔊", label: "Audio" },
];

function inputIsLinked(node, inputName) {
  const input = node.inputs?.find((item) => item?.name === inputName);
  return Boolean(input && (input.link != null || input.links?.length));
}

function availableAssets(node) {
  const assets = [];
  for (const type of ASSET_TYPES) {
    for (let slot = 1; slot <= (type.label === "Picture" ? 9 : 3); slot += 1) {
      const name = `${type.prefix}${slot}`;
      if (inputIsLinked(node, name)) {
        assets.push({
          token: `@${type.tokenPrefix ?? type.prefix}${slot}`,
          title: `${type.icon} ${type.label} ${slot}`,
          detail: name,
        });
      }
    }
  }
  return assets;
}

function activeAtToken(textarea) {
  const cursor = textarea.selectionStart ?? 0;
  const beforeCursor = textarea.value.slice(0, cursor);
  const match = beforeCursor.match(/@([a-z0-9_]*)$/i);
  if (!match) return null;
  return { start: cursor - match[0].length, end: cursor, query: match[1].toLowerCase() };
}

function createMenu() {
  const menu = document.createElement("div");
  Object.assign(menu.style, {
    position: "fixed",
    zIndex: "100000",
    display: "none",
    minWidth: "230px",
    maxHeight: "220px",
    overflowY: "auto",
    padding: "4px",
    border: "1px solid var(--border-color, #555)",
    borderRadius: "7px",
    background: "var(--comfy-menu-bg, #202020)",
    color: "var(--input-text, #f1f1f1)",
    boxShadow: "0 8px 24px rgba(0, 0, 0, .45)",
    fontFamily: "sans-serif",
    fontSize: "12px",
  });
  document.body.appendChild(menu);
  return menu;
}

function editableTextElement(widget) {
  const element = widget?.inputEl ?? widget?.element;
  return element instanceof HTMLTextAreaElement || element instanceof HTMLInputElement ? element : null;
}

function attachAutocomplete(assetNode, editableNode, widget) {
  const textarea = editableTextElement(widget);
  const bindingKey = `qwen38AutocompleteAttached${assetNode.id}`;
  if (!textarea || textarea.dataset[bindingKey]) return;
  textarea.dataset[bindingKey] = "true";

  const menu = createMenu();
  let choices = [];
  let selectedIndex = 0;
  let replacement = null;

  const hide = () => {
    choices = [];
    replacement = null;
    menu.style.display = "none";
    menu.replaceChildren();
  };

  const commit = (choice, range = replacement) => {
    if (!choice || !range) return;
    const value = `${textarea.value.slice(0, range.start)}${choice.token}${textarea.value.slice(range.end)}`;
    const cursor = range.start + choice.token.length;
    const sync = () => {
      const elementPrototype = textarea instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const nativeSetter = Object.getOwnPropertyDescriptor(elementPrototype, "value")?.set;
      if (nativeSetter) nativeSetter.call(textarea, value);
      else textarea.value = value;
      textarea.selectionStart = cursor;
      textarea.selectionEnd = cursor;
      widget.value = value;
      if (widget.inputEl && widget.inputEl !== textarea) widget.inputEl.value = value;
      if (widget.element && widget.element !== textarea) widget.element.value = value;
      editableNode.setDirtyCanvas?.(true, true);
    };
    sync();
    widget.callback?.(value);
    textarea.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
    textarea.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
    requestAnimationFrame(sync);
    hide();
    textarea.focus();
  };

  const render = () => {
    menu.replaceChildren();
    choices.forEach((choice, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `${choice.title}  —  ${choice.token}`;
      Object.assign(button.style, {
        display: "block",
        width: "100%",
        border: "0",
        borderRadius: "4px",
        padding: "7px 9px",
        textAlign: "left",
        cursor: "pointer",
        color: "inherit",
        background: index === selectedIndex ? "var(--comfy-input-bg, #3b3b3b)" : "transparent",
      });
      button.addEventListener("mouseenter", () => {
        selectedIndex = index;
        render();
      });
      const range = replacement && { ...replacement };
      button.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        commit(choice, range);
      });
      menu.appendChild(button);
    });
  };

  const show = () => {
    replacement = activeAtToken(textarea);
    if (!replacement) {
      hide();
      return;
    }
    const query = replacement.query;
    choices = availableAssets(assetNode).filter((choice) => choice.token.slice(1).toLowerCase().includes(query));
    if (!choices.length) {
      hide();
      return;
    }
    selectedIndex = Math.min(selectedIndex, choices.length - 1);
    const rect = textarea.getBoundingClientRect();
    menu.style.left = `${Math.min(rect.left, window.innerWidth - 250)}px`;
    menu.style.top = `${Math.min(rect.bottom + 4, window.innerHeight - 230)}px`;
    menu.style.display = "block";
    render();
  };

  textarea.addEventListener("input", show);
  textarea.addEventListener("focus", show);
  textarea.addEventListener("blur", () => setTimeout(hide, 150));
  textarea.addEventListener("keydown", (event) => {
    if (menu.style.display === "none") return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      selectedIndex = (selectedIndex + 1) % choices.length;
      render();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      selectedIndex = (selectedIndex - 1 + choices.length) % choices.length;
      render();
    } else if (event.key === "Enter" || event.key === "Tab") {
      event.preventDefault();
      commit(choices[selectedIndex]);
    } else if (event.key === "Escape") {
      event.preventDefault();
      hide();
    }
  });

  const originalOnRemoved = assetNode.onRemoved;
  assetNode.onRemoved = function (...args) {
    menu.remove();
    return originalOnRemoved?.apply(this, args);
  };
}

function linkedTextSource(node) {
  const input = node.inputs?.find((item) => item?.name === EXTERNAL_BRIEF_INPUT);
  const linkId = input?.link ?? input?.links?.[0];
  if (linkId == null) return null;
  const link = node.graph?.links?.[linkId] ?? app.graph?.links?.[linkId];
  if (!link) return null;
  return node.graph?.getNodeById?.(link.origin_id) ?? app.graph?.getNodeById?.(link.origin_id) ?? null;
}

function externalTextWidget(sourceNode) {
  const widgets = sourceNode?.widgets ?? [];
  const preferredNames = /^(brief|text|string|value|prompt)$/i;
  return (
    widgets.find((widget) => preferredNames.test(widget?.name ?? "") && editableTextElement(widget))
    ?? widgets.find((widget) => editableTextElement(widget))
    ?? null
  );
}

function attachExternalBriefAutocomplete(assetNode) {
    const sourceNode = linkedTextSource(assetNode);
    const widget = externalTextWidget(sourceNode);
    if (sourceNode && widget) attachAutocomplete(assetNode, sourceNode, widget);
}

function normalizeDirectorGroupSpecs(value) {
  let specs = value;
  // ComfyUI merges one UI result as [[{...}, {...}]].
  while (Array.isArray(specs) && specs.length === 1 && Array.isArray(specs[0])) specs = specs[0];
  return Array.isArray(specs) ? specs.filter((item) => item && typeof item === "object") : [];
}

function qwen38PositiveSlots(value, max = 9) {
  const slots = Array.isArray(value) ? value : [];
  return [...new Set(slots.map((slot) => Number(slot)).filter(
    (slot) => Number.isInteger(slot) && slot >= 1 && slot <= max,
  ))];
}

function isLongDirectorPlanner(node) {
  return (node?.comfyClass ?? node?.type) === "Qwen38LongVideoDirectorPlanner";
}

function savedWidgetNames(workflowNode) {
  const names = [];
  for (const input of workflowNode?.inputs ?? []) {
    const name = input?.widget?.name;
    if (!name) continue;
    names.push(name);
    // Comfy serializes the seed's after-generate control as one extra widget
    // value, even though it has no separate input socket in workflow JSON.
    if (name === "seed") names.push("control_after_generate");
  }
  return names;
}

const LEGACY_WIDGET_ORDERS = {
  Qwen38LocalPromptRefiner: [
    "brief", "engine", "model_name", "mmproj_name", "vision_handler", "context_length", "gpu_layers",
    "temperature", "top_p", "max_tokens", "seed", "control_after_generate", "keep_model_loaded",
    "lora_preset_1", "lora_preset_2", "lora_preset_3", "video_frame_samples", "max_reference_media_seconds",
    "reference_text", "manual_lora_trigger_words",
  ],
  Qwen38LongVideoDirectorPlanner: [
    "brief", "director_mode", "total_duration_seconds", "segment_duration_seconds", "example_preset",
    "model_name", "mmproj_name", "vision_handler", "context_length", "gpu_layers", "temperature", "top_p",
    "max_tokens", "seed", "control_after_generate", "keep_model_loaded", "lora_preset_1", "lora_preset_2",
    "lora_preset_3", "video_frame_samples", "reference_text", "manual_lora_trigger_words",
  ],
};

function hasLegacyWidgetValueShift(nodeName, values) {
  if (nodeName === "Qwen38LocalPromptRefiner") {
    // Before example_preset was added, index 2 was model_name. A GGUF at that
    // position proves a positional shift even if an autosave has newer inputs.
    return typeof values?.[2] === "string" && /\.(gguf|ggml)(?:$|[\\/])/i.test(values[2]);
  }
  // Before planning_quality was added, index 2 was total_duration_seconds.
  const shiftedTotal = values?.[2];
  return typeof shiftedTotal === "number"
    || (typeof shiftedTotal === "string" && /^\d+(?:\.\d+)?$/.test(shiftedTotal));
}

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  const element = widget.inputEl ?? widget.element;
  if (element && "value" in element) element.value = value;
  widget.callback?.(value);
}

function migrateLegacyWidgetLayout(node, workflowNode, nodeName) {
  const oldValues = workflowNode?.widgets_values;
  if (!Array.isArray(oldValues) || !Array.isArray(workflowNode?.inputs)) return false;

  const savedNames = savedWidgetNames(workflowNode);
  const savedNameSet = new Set(savedNames);
  const missingCurrentControls = nodeName === "Qwen38LocalPromptRefiner"
    ? !savedNameSet.has("example_preset")
    : !savedNameSet.has("planning_quality") || !savedNameSet.has("director_execution");
  const needsMigration = missingCurrentControls || hasLegacyWidgetValueShift(nodeName, oldValues);
  if (!needsMigration) return false;

  const sourceNames = missingCurrentControls ? savedNames : LEGACY_WIDGET_ORDERS[nodeName];
  if (!sourceNames) return false;

  const valueByName = new Map();
  sourceNames.forEach((name, index) => {
    if (index < oldValues.length) valueByName.set(name, oldValues[index]);
  });
  for (const widget of node.widgets ?? []) {
    if (valueByName.has(widget.name)) setWidgetValue(widget, valueByName.get(widget.name));
  }

  // New controls did not exist in the saved node, so positional restoration
  // leaves them holding unrelated old values. Reset only those new controls.
  if (nodeName === "Qwen38LocalPromptRefiner") {
    setWidgetValue(node.widgets?.find((widget) => widget?.name === "example_preset"), "None");
  } else {
    setWidgetValue(node.widgets?.find((widget) => widget?.name === "planning_quality"), "Balanced (recommended)");
    setWidgetValue(node.widgets?.find((widget) => widget?.name === "director_execution"), "Preview plan only (Director blocked)");
  }

  // Persist the corrected current-order values the next time Comfy saves the
  // workflow, so a later restart does not shift video_frame_samples, reference
  // text, or literal LoRA trigger words again.
  workflowNode.widgets_values = (node.widgets ?? []).map((widget) => widget.value);
  node.properties ??= {};
  node.properties.qwen38WidgetLayout = "2026-09-12";
  node.setDirtyCanvas?.(true, true);
  return true;
}

function refreshDirectorGroupPreviews() {
  const graph = app.graph ?? app.canvas?.graph;
  for (const graphNode of graph?._nodes ?? graph?.nodes ?? []) {
    graphNode?._minimaxEditor?.syncExternalGroupsTimeline?.();
  }
}

// Native Director materialization -------------------------------------------------
//
// A Qwen long-plan output is already a valid MMX_DIR_GROUP payload at execution
// time, but one planner node can contain many generated groups.  The Director's
// own frontend intentionally renders its editable external cards from *native*
// Group nodes in the graph.  These helpers turn a reviewed Qwen plan into that
// native graph shape without changing the Director extension:
//
//   Qwen planner (preview only) → [native Group × N] → Combine → Director
//
// The materialized nodes become the Director's source of truth.  Therefore the
// planner is set to Never after materialization; queueing Director cannot spend
// another Qwen inference.  Re-enable the planner, queue a new preview, then
// materialize again when the story needs revision.
const DIRECTOR_NODE_TYPES = new Set(["MiniMaxH3Director", "ComfyMiniMaxH3Director"]);
const DIRECTOR_R2V_GROUP_TYPE = "MiniMaxH3DirectorGroupReferenceToVideo";
const DIRECTOR_I2V_GROUP_TYPE = "MiniMaxH3DirectorGroupImageToVideo";
const DIRECTOR_COMBINE_TYPE = "MiniMaxH3DirectorGroupsCombine";
const MATERIALIZED_BY_PROPERTY = "qwen38DirectorMaterializedBy";

function qwen38Graph() {
  return app.graph ?? app.canvas?.graph ?? null;
}

function qwen38NodeClass(node) {
  return String(node?.comfyClass ?? node?.type ?? "");
}

function qwen38GraphLink(graph, linkId) {
  if (!graph || linkId == null) return null;
  const links = graph.links;
  let link = links?.[linkId];
  if (!link && typeof links?.find === "function") {
    link = links.find((candidate) => candidate && (candidate.id === linkId || candidate[0] === linkId));
  }
  if (!link) return null;
  return {
    originId: link.origin_id ?? link[1],
    originSlot: link.origin_slot ?? link[2],
    targetId: link.target_id ?? link[3],
    targetSlot: link.target_slot ?? link[4],
  };
}

function qwen38InputIndex(node, name) {
  return (node?.inputs ?? []).findIndex((input) => input?.name === name);
}

function qwen38OutputIndex(node, name) {
  return (node?.outputs ?? []).findIndex((output) => output?.name === name);
}

function qwen38InputSource(graph, node, inputName) {
  const index = qwen38InputIndex(node, inputName);
  const input = index >= 0 ? node.inputs[index] : null;
  const link = qwen38GraphLink(graph, input?.link);
  if (!link) return null;
  const source = graph.getNodeById?.(link.originId);
  if (!source || !Number.isInteger(Number(link.originSlot))) return null;
  return { node: source, outputIndex: Number(link.originSlot) };
}

function qwen38FindDirector(graph, planner) {
  const groupOutput = qwen38OutputIndex(planner, "director_groups");
  const output = groupOutput >= 0 ? planner?.outputs?.[groupOutput] : null;
  for (const linkId of output?.links ?? []) {
    const link = qwen38GraphLink(graph, linkId);
    const target = graph.getNodeById?.(link?.targetId);
    if (DIRECTOR_NODE_TYPES.has(qwen38NodeClass(target))) return target;
  }
  return (graph?._nodes ?? graph?.nodes ?? []).find((node) => DIRECTOR_NODE_TYPES.has(qwen38NodeClass(node))) ?? null;
}

function qwen38CreateGraphNode(graph, type, position) {
  const node = globalThis.LiteGraph?.createNode?.(type);
  if (!node) {
    throw new Error(`找不到导演台原生节点：${type}。请确认 ComfyUI_MiniMaxH3_Director 已安装并刷新网页。`);
  }
  graph.add(node);
  if (Array.isArray(position)) node.pos = position;
  node.properties ??= {};
  return node;
}

function qwen38SetNodeWidget(node, name, value) {
  const widget = (node?.widgets ?? []).find((candidate) => candidate?.name === name);
  if (!widget) throw new Error(`原生 Director Group 缺少 ${name} 控件。请更新导演台插件后重试。`);
  setWidgetValue(widget, value);
}

function qwen38FindOrAddInput(node, preferredName, type) {
  let index = qwen38InputIndex(node, preferredName);
  if (index >= 0) return index;
  // Autogrow nodes may expose the legacy leaf name instead of the nested name
  // (`ref_image_0` vs `ref_images.ref_image_0`) on older ComfyUI frontends.
  const leaf = String(preferredName).split(".").pop();
  index = (node?.inputs ?? []).findIndex((input) => String(input?.name ?? "").endsWith(`.${leaf}`) || input?.name === leaf);
  if (index >= 0) return index;
  node.addInput?.(preferredName, type);
  index = qwen38InputIndex(node, preferredName);
  if (index < 0) throw new Error(`无法创建原生 Group 输入：${preferredName}`);
  return index;
}

function qwen38ConnectSource(graph, source, target, targetInputName, type) {
  if (!source) return false;
  const inputIndex = qwen38FindOrAddInput(target, targetInputName, type);
  source.node.connect?.(source.outputIndex, target, inputIndex);
  return true;
}

function qwen38DirectorPortForSpecs(specs, planner) {
  if (specs.some((spec) => String(spec?.family ?? "") === "i2v")) return "i2v_groups";
  const mode = String((planner?.widgets ?? []).find((widget) => widget?.name === "director_mode")?.value ?? "");
  return /i2v_groups|fl2v/i.test(mode) ? "i2v_groups" : "r2v_groups";
}

function qwen38DisconnectInput(node, name) {
  const index = qwen38InputIndex(node, name);
  if (index >= 0 && node?.inputs?.[index]?.link != null) node.disconnectInput?.(index);
}

function qwen38SetDirectorTaskForPort(director, port) {
  // These are the Director's canonical task labels. The backend normalizes the
  // label back to r2v / fl2v, while the widget callback refreshes its timeline
  // UI before we apply Qwen's per-segment continuity flags.
  const task = port === "i2v_groups"
    ? "首尾帧生视频(First-Last Frame)"
    : "参考主体生视频(Reference to Video)";
  const widget = (director?.widgets ?? []).find((candidate) => candidate?.name === "task_type");
  if (widget) setWidgetValue(widget, task);
}

function qwen38DisconnectPlannerGroupOutput(planner) {
  const outputIndex = qwen38OutputIndex(planner, "director_groups");
  if (outputIndex >= 0 && planner?.outputs?.[outputIndex]?.links?.length) {
    planner.disconnectOutput?.(outputIndex);
  }
}

function qwen38RemovePriorMaterialization(graph, planner) {
  const owned = (graph?._nodes ?? graph?.nodes ?? []).filter(
    (node) => String(node?.properties?.[MATERIALIZED_BY_PROPERTY] ?? "") === String(planner?.id),
  );
  if (!owned.length) return;
  const approved = globalThis.confirm?.(
    `此 Qwen 节点已经生成了 ${owned.length} 个原生 Director Group / Combine 节点。\n\n替换它们为当前预览方案？你对这些自动生成节点做的手工编辑会被替换。`,
  );
  if (approved === false) throw new Error("已取消：保留现有原生 Director Groups。 ");
  for (const node of owned) graph.remove?.(node);
}

function qwen38ApplyDirectorContinuity(director, specs, attempts = 0) {
  const editor = director?._minimaxEditor;
  const timeline = editor?.timeline;
  const rows = timeline?.segments ?? timeline?.shots;
  if (!Array.isArray(rows) || rows.length !== specs.length) {
    if (attempts < 12) {
      setTimeout(() => qwen38ApplyDirectorContinuity(director, specs, attempts + 1), 80);
    }
    return;
  }
  timeline.output ??= {};
  timeline.output.continuityEnabled = specs.some(
    (spec, index) => index > 0 && Boolean(spec?.continuity_from_prev),
  );
  rows.forEach((row, index) => {
    row.continuityFromPrev = index > 0 && Boolean(specs[index]?.continuity_from_prev);
  });
  editor.commit?.(false, { syncTimeline: true });
  editor.updateSelectionUI?.();
  editor.scheduleRender?.();
}

function qwen38MaterializeR2vGroup(graph, planner, spec, position) {
  const group = qwen38CreateGraphNode(graph, DIRECTOR_R2V_GROUP_TYPE, position);
  qwen38SetNodeWidget(group, "prompt", String(spec.prompt ?? ""));
  qwen38SetNodeWidget(group, "duration_sec", Number(spec.duration_sec) || 5);
  group.title = `Qwen Director Group ${Number(spec.index) || 0}`;
  group.properties[MATERIALIZED_BY_PROPERTY] = String(planner.id);
  group.properties.qwen38DirectorSegmentIndex = Number(spec.index) || 0;

  const assignments = [
    ["pictures", "reference_image_", "ref_images.ref_image_", "IMAGE"],
    ["videos", "reference_video_frames_", "ref_videos.ref_video_", "IMAGE"],
    ["video_audios", "reference_video_audio_", "ref_video_audios.ref_video_audio_", "AUDIO"],
    ["audios", "reference_audio_", "ref_audios.ref_audio_", "AUDIO"],
  ];
  for (const [specKey, plannerPrefix, groupPrefix, type] of assignments) {
    for (const slot of qwen38PositiveSlots(spec?.[specKey], 9)) {
      const source = qwen38InputSource(graph, planner, `${plannerPrefix}${slot}`);
      if (!source) continue;
      qwen38ConnectSource(graph, source, group, `${groupPrefix}${slot - 1}`, type);
    }
  }
  return group;
}

function qwen38MaterializeI2vGroup(graph, planner, spec, position) {
  const group = qwen38CreateGraphNode(graph, DIRECTOR_I2V_GROUP_TYPE, position);
  qwen38SetNodeWidget(group, "prompt", String(spec.prompt ?? ""));
  qwen38SetNodeWidget(group, "duration_sec", Number(spec.duration_sec) || 5);
  group.title = `Qwen Director Group ${Number(spec.index) || 0}`;
  group.properties[MATERIALIZED_BY_PROPERTY] = String(planner.id);
  group.properties.qwen38DirectorSegmentIndex = Number(spec.index) || 0;
  const firstSlot = Number(spec?.first_picture);
  const lastSlot = Number(spec?.last_picture);
  if (Number.isInteger(firstSlot) && firstSlot >= 1) {
    qwen38ConnectSource(
      graph,
      qwen38InputSource(graph, planner, `reference_image_${firstSlot}`),
      group,
      "first_frame",
      "IMAGE",
    );
  }
  if (Number.isInteger(lastSlot) && lastSlot >= 1) {
    qwen38ConnectSource(
      graph,
      qwen38InputSource(graph, planner, `reference_image_${lastSlot}`),
      group,
      "last_frame",
      "IMAGE",
    );
  }
  return group;
}

function qwen38ConnectGroupsToCombine(graph, groups, combine) {
  groups.forEach((group, index) => {
    const inputIndex = qwen38FindOrAddInput(combine, `groups.group_${index}`, "MMX_DIR_GROUP");
    const outputIndex = qwen38OutputIndex(group, "group");
    if (outputIndex < 0) throw new Error("原生 Director Group 没有 group 输出。请更新导演台插件后重试。");
    group.connect?.(outputIndex, combine, inputIndex);
  });
}

function qwen38MaterializeNativeDirectorGroups(planner) {
  const graph = qwen38Graph();
  const specs = normalizeDirectorGroupSpecs(
    planner?._qwen38DirectorGroupSpecs ?? planner?.properties?.qwen38DirectorGroupSpecs,
  );
  if (!graph) throw new Error("ComfyUI 画布尚未就绪。请刷新页面后重试。");
  if (!specs.length) throw new Error("尚无可生成的 Qwen 分镜。请先以 Preview plan only 运行一次规划节点。");
  const director = qwen38FindDirector(graph, planner);
  if (!director) throw new Error("未找到 MiniMax H3 Director 节点。请先将规划节点放入含导演台的工作流。");
  const port = qwen38DirectorPortForSpecs(specs, planner);
  const groupType = port === "i2v_groups" ? DIRECTOR_I2V_GROUP_TYPE : DIRECTOR_R2V_GROUP_TYPE;
  if (!globalThis.LiteGraph?.registered_node_types?.[groupType]
    || !globalThis.LiteGraph?.registered_node_types?.[DIRECTOR_COMBINE_TYPE]) {
    throw new Error("导演台原生 Group / Combine 节点尚未注册。请确认导演台已加载，然后按 Ctrl+R 刷新 ComfyUI 页面。");
  }

  qwen38RemovePriorMaterialization(graph, planner);
  const plannerX = Number(planner?.pos?.[0]) || 0;
  const plannerY = Number(planner?.pos?.[1]) || 0;
  const plannerWidth = Number(planner?.size?.[0]) || 420;
  const groupX = plannerX + plannerWidth + 90;
  const groupY = plannerY;
  const groupGap = 250;
  const created = [];
  try {
    for (const [ordinal, spec] of specs.entries()) {
      const position = [groupX, groupY + ordinal * groupGap];
      created.push(
        port === "i2v_groups"
          ? qwen38MaterializeI2vGroup(graph, planner, spec, position)
          : qwen38MaterializeR2vGroup(graph, planner, spec, position),
      );
    }
    const combine = qwen38CreateGraphNode(
      graph,
      DIRECTOR_COMBINE_TYPE,
      [groupX + 470, groupY + Math.max(0, (created.length - 1) * groupGap / 2)],
    );
    combine.title = "Qwen Director Groups Combine";
    combine.properties[MATERIALIZED_BY_PROPERTY] = String(planner.id);
    qwen38ConnectGroupsToCombine(graph, created, combine);

    // Replace only the relevant external port. A planner's direct output must
    // be disconnected before Combine is wired, otherwise Director can retain a
    // stale direct plan link in an old workflow.
    qwen38DisconnectPlannerGroupOutput(planner);
    qwen38DisconnectInput(director, port);
    const combineOutput = qwen38OutputIndex(combine, "groups");
    const directorInput = qwen38InputIndex(director, port);
    if (combineOutput < 0 || directorInput < 0) {
      throw new Error(`找不到 Director.${port}，请检查导演台版本与模式。`);
    }
    combine.connect?.(combineOutput, director, directorInput);
    qwen38SetDirectorTaskForPort(director, port);

    // Avoid a second local-model inference on the Director queue. The node can
    // be switched back to Always when the user wants to plan a revised brief.
    planner.mode = globalThis.LiteGraph?.NEVER ?? 2;
    planner.properties ??= {};
    planner.properties.qwen38DirectorMaterialized = {
      version: 1,
      port,
      count: created.length,
      generatedAt: new Date().toISOString(),
    };
    graph.setDirtyCanvas?.(true, true);
    planner.setDirtyCanvas?.(true, true);
    requestAnimationFrame(() => qwen38ApplyDirectorContinuity(director, specs));
    globalThis.alert?.(
      `已生成 ${created.length} 个导演台原生 Group 和 1 个 Combine，并连接到 Director.${port}。\n\nQwen 规划节点已设为 Never，因此下一次 Queue 会直接运行导演台。需要重写分镜时，将 Qwen 节点切回 Always，再运行 Preview。`,
    );
  } catch (error) {
    for (const node of created) graph.remove?.(node);
    throw error;
  }
}

function qwen38AddMaterializeControls(planner) {
  if (planner?._qwen38DirectorMaterializeControlsAdded) return;
  planner._qwen38DirectorMaterializeControlsAdded = true;
  const materialize = () => {
    try {
      qwen38MaterializeNativeDirectorGroups(planner);
    } catch (error) {
      globalThis.alert?.(`无法生成原生 Director Groups：${error?.message ?? error}`);
      console.error("Qwen38 native Director Group materialization failed", error);
    }
  };
  // A non-serialized button cannot shift the backend widget order in existing
  // workflows. The same operation is also in the right-click menu for compact
  // node layouts.
  planner.addWidget?.("button", "生成原生 Director Groups", null, materialize, { serialize: false });
  const originalMenu = planner.getExtraMenuOptions;
  planner.getExtraMenuOptions = function (...args) {
    const options = originalMenu?.apply(this, args) ?? [];
    options.push({ content: "生成原生 Director Groups", callback: materialize });
    return options;
  };
}

app.registerExtension({
  name: "Qwen38LocalPromptRefiner.ReferenceAutocomplete",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) return;
    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (workflowNode) {
      const configured = originalOnConfigure?.apply(this, arguments);
      migrateLegacyWidgetLayout(this, workflowNode, nodeData.name);
      return configured;
    };
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      let attempts = 0;
      const attach = () => {
        const widget = this.widgets?.find((item) => item?.name === WIDGET_NAME);
        if (widget?.inputEl ?? widget?.element) {
          attachAutocomplete(this, this, widget);
        }
        attachExternalBriefAutocomplete(this);
        if (attempts++ < 20) setTimeout(attach, 100);
      };
      attach();

      if (isLongDirectorPlanner(this)) {
        qwen38AddMaterializeControls(this);
        const restoredSpecs = normalizeDirectorGroupSpecs(this.properties?.qwen38DirectorGroupSpecs);
        if (restoredSpecs.length) this._qwen38DirectorGroupSpecs = restoredSpecs;
        const originalOnExecuted = this.onExecuted;
        this.onExecuted = function (message) {
          const executedResult = originalOnExecuted?.apply(this, arguments);
          const specs = normalizeDirectorGroupSpecs(message?.qwen38_director_groups);
          if (specs.length) {
            this._qwen38DirectorGroupSpecs = specs;
            this.properties ??= {};
            this.properties.qwen38DirectorGroupSpecs = specs;
            this.setDirtyCanvas?.(true, true);
            queueMicrotask(refreshDirectorGroupPreviews);
          }
          return executedResult;
        };
      }

      const originalOnConnectionsChange = this.onConnectionsChange;
      this.onConnectionsChange = function (...connectionArgs) {
        const connectionResult = originalOnConnectionsChange?.apply(this, connectionArgs);
        requestAnimationFrame(() => attachExternalBriefAutocomplete(this));
        return connectionResult;
      };
      return result;
    };
  },
});
