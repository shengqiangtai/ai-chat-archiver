const CLAUDE_SELECTORS = {
  messageNodes: [
    "[data-testid='user-message']",
    "[data-testid='assistant-message']",
    ".human-turn",
    ".ai-turn",
  ].join(", "),
  model: "button[data-testid='model-selector'], [data-testid='chat-model-selector']",
};

function claudeRole(node) {
  const testId = (node.getAttribute("data-testid") || "").toLowerCase();
  const className = (node.className || "").toString().toLowerCase();
  if (testId.includes("user") || className.includes("human")) return "user";
  if (testId.includes("assistant") || className.includes("ai-turn")) return "assistant";
  return "";
}

class ClaudeExtractor extends BaseExtractor {
  getPlatformName() {
    return "Claude";
  }

  extract() {
    const nodes = Array.from(document.querySelectorAll(CLAUDE_SELECTORS.messageNodes));
    const messages = [];

    for (const node of nodes) {
      const role = claudeRole(node);
      if (!role) continue;
      const content = cleanText(node);
      if (!content) continue;
      messages.push({ role, content, time: nowISO() });
    }

    const title = getPageTitle().replace(/\s*-\s*Claude$/i, "").trim() || "Untitled Chat";
    const model = cleanText(document.querySelector(CLAUDE_SELECTORS.model)) || null;

    return {
      title,
      model,
      url: getPageURL(),
      messages,
    };
  }
}

window.__aiArchiverExtractor = new ClaudeExtractor();
