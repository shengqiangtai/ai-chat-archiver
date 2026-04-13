const CHATGPT_SELECTORS = {
  messageNodes: "[data-message-author-role], article[data-testid*='conversation-turn'] [data-message-author-role]",
  model: "button[data-testid='model-switcher-dropdown-button'], [data-testid='conversation-model']",
};

function isVisible(el) {
  if (!el) return false;
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

class ChatGPTExtractor extends BaseExtractor {
  getPlatformName() {
    return "ChatGPT";
  }

  extract() {
    const nodes = Array.from(document.querySelectorAll(CHATGPT_SELECTORS.messageNodes)).filter(isVisible);
    const messages = [];

    for (const node of nodes) {
      const role = (node.getAttribute("data-message-author-role") || "").toLowerCase();
      if (role !== "user" && role !== "assistant") continue;
      const content = cleanText(node);
      if (!content) continue;
      messages.push({ role, content, time: nowISO() });
    }

    const modelNode = document.querySelector(CHATGPT_SELECTORS.model);
    const title = getPageTitle().replace(/\s*-\s*ChatGPT$/i, "").trim() || "Untitled Chat";

    return {
      title,
      model: cleanText(modelNode) || null,
      url: getPageURL(),
      messages,
    };
  }
}

window.__aiArchiverExtractor = new ChatGPTExtractor();
