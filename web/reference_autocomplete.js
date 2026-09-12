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
