import { app } from "../../scripts/app.js";

const NODE_NAME = "Qwen38LocalPromptRefiner";
const WIDGET_NAME = "brief";
const ASSET_TYPES = [
  { prefix: "reference_image_", icon: "🖼", label: "Picture" },
  { prefix: "reference_video_", icon: "🎞", label: "Video" },
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
          token: `@${name}`,
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

function attachAutocomplete(node, widget) {
  const textarea = widget?.inputEl ?? widget?.element;
  if (!(textarea instanceof HTMLTextAreaElement) || textarea.dataset.qwen38AutocompleteAttached) return;
  textarea.dataset.qwen38AutocompleteAttached = "true";

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
      const nativeSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      if (nativeSetter) nativeSetter.call(textarea, value);
      else textarea.value = value;
      textarea.selectionStart = cursor;
      textarea.selectionEnd = cursor;
      widget.value = value;
      if (widget.inputEl && widget.inputEl !== textarea) widget.inputEl.value = value;
      if (widget.element && widget.element !== textarea) widget.element.value = value;
      node.setDirtyCanvas?.(true, true);
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
    choices = availableAssets(node).filter((choice) => choice.token.slice(1).toLowerCase().includes(query));
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

  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    menu.remove();
    return originalOnRemoved?.apply(this, args);
  };
}

app.registerExtension({
  name: "Qwen38LocalPromptRefiner.ReferenceAutocomplete",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      let attempts = 0;
      const attach = () => {
        const widget = this.widgets?.find((item) => item?.name === WIDGET_NAME);
        if (widget?.inputEl ?? widget?.element) {
          attachAutocomplete(this, widget);
          return;
        }
        if (attempts++ < 20) setTimeout(attach, 100);
      };
      attach();
      return result;
    };
  },
});
