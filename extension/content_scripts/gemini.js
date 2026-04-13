const GEMINI_SELECTORS = {
  user: "user-query, [data-testid*='user-query'], [class*='query-text']",
  assistant: "model-response, [data-testid*='model-response'], [class*='response-container']",
  all: "user-query, model-response, [data-testid*='user-query'], [data-testid*='model-response'], [class*='query-text'], [class*='response-container']",
  model: "button[aria-label*='Model'], [data-test-id='model-selector']",
};

class GeminiExtractor extends BaseExtractor {
  getPlatformName() {
    return "Gemini";
  }

  extract() {
    const messages = [];
    const nodes = Array.from(document.querySelectorAll(GEMINI_SELECTORS.all));

    for (const node of nodes) {
      const content = cleanText(node);
      if (!content) continue;

      const isUser =
        node.matches(GEMINI_SELECTORS.user) &&
        !node.matches(GEMINI_SELECTORS.assistant);
      const role = isUser ? "user" : "assistant";
      messages.push({ role, content, time: nowISO() });
    }

    const title = getPageTitle().replace(/\s*-\s*Gemini$/i, "").trim() || "Untitled Chat";
    const model = cleanText(document.querySelector(GEMINI_SELECTORS.model)) || null;

    return {
      title,
      model,
      url: getPageURL(),
      messages,
    };
  }
}

window.__aiArchiverExtractor = new GeminiExtractor();
