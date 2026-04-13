const DEEPSEEK_SELECTORS = {
  messageNodes: [
    ".dad65929",
    ".f9bf2b70",
    "[data-role='user']",
    "[data-role='assistant']",
    "[class*='user-message']",
    "[class*='assistant-message']",
  ].join(", "),
  model: "[class*='model-name'], [class*='chat-model'], button[aria-haspopup='menu']",
};

function deepseekRole(node) {
  const cls = (node.className || "").toString().toLowerCase();
  const dataRole = (node.getAttribute("data-role") || "").toLowerCase();
  if (dataRole.includes("user") || cls.includes("user")) return "user";
  if (dataRole.includes("assistant") || cls.includes("assistant") || cls.includes("bot")) return "assistant";
  return "";
}

class DeepSeekExtractor extends BaseExtractor {
  getPlatformName() {
    return "DeepSeek";
  }

  extract() {
    const nodes = Array.from(document.querySelectorAll(DEEPSEEK_SELECTORS.messageNodes));
    const messages = [];

    for (const node of nodes) {
      const role = deepseekRole(node);
      if (!role) continue;
      const content = cleanText(node);
      if (!content) continue;
      messages.push({ role, content, time: nowISO() });
    }

    const title = getPageTitle().replace(/\s*-\s*DeepSeek$/i, "").trim() || "Untitled Chat";
    const model = cleanText(document.querySelector(DEEPSEEK_SELECTORS.model)) || null;

    return {
      title,
      model,
      url: getPageURL(),
      messages,
    };
  }
}

window.__aiArchiverExtractor = new DeepSeekExtractor();
