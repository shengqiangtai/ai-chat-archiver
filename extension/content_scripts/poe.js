const POE_SELECTORS = {
  threadRoots: [
    "main",
    "[class*='ChatPage']",
    "[class*='Conversation']",
    "[class*='MessagesView']",
    "[class*='ChatMessagesView']",
  ].join(", "),
  messageRows: [
    "[data-testid*='message']",
    "[class*='MessageRow']",
    "[class*='messageRow']",
    "[class*='Message_message']",
    "[class*='ChatMessage']",
    "[class*='Message_']",
  ].join(", "),
  userRow: [
    "[class*='humanMessage']",
    "[class*='HumanMessage']",
    "[class*='userMessage']",
    "[class*='UserMessage']",
    "[data-message-author='human']",
    "[data-author='human']",
    "[data-role='user']",
  ].join(", "),
  assistantRow: [
    "[class*='botMessage']",
    "[class*='BotMessage']",
    "[class*='assistantMessage']",
    "[class*='AssistantMessage']",
    "[data-message-author='bot']",
    "[data-author='bot']",
    "[data-role='assistant']",
  ].join(", "),
  messageContent: [
    "[class*='Markdown']",
    "[class*='markdown']",
    "[class*='MessageContent']",
    "[class*='messageContent']",
    "[class*='Message_text']",
    "[class*='messageText']",
    "[data-testid*='message-content']",
  ].join(", "),
  model: [
    "[class*='BotHeader'] [class*='BotName']",
    "[class*='botHeader'] [class*='botName']",
    "[data-testid*='bot-name']",
    "[class*='ModelName']",
  ].join(", "),
};

function poeVisible(el) {
  if (!el) return false;
  const style = window.getComputedStyle(el);
  if (style.display === "none" || style.visibility === "hidden") return false;
  if (el.getAttribute("aria-hidden") === "true") return false;
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

function looksLikeRoleLabel(line) {
  const text = (line || "").trim();
  if (!text) return true;
  const lower = text.toLowerCase();
  if (lower === "assistant" || lower === "user" || lower === "you" || lower === "poe") return true;
  if (/^[a-z0-9][a-z0-9.+-]{1,40}$/i.test(text) && /[.-]/.test(text)) return true;
  return false;
}

function normalizePoeContent(raw) {
  const lines = (raw || "")
    .replace(/\u00a0/g, " ")
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  while (lines.length > 1 && looksLikeRoleLabel(lines[0])) {
    lines.shift();
  }

  return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function getPoeRole(row) {
  if (row.matches(POE_SELECTORS.userRow) || row.querySelector(POE_SELECTORS.userRow)) return "user";
  if (row.matches(POE_SELECTORS.assistantRow) || row.querySelector(POE_SELECTORS.assistantRow)) return "assistant";

  const cls = (row.className || "").toString().toLowerCase();
  if (cls.includes("human") || cls.includes("user")) return "user";
  if (cls.includes("bot") || cls.includes("assistant")) return "assistant";
  return "";
}

function pickThreadRoot() {
  const roots = Array.from(document.querySelectorAll(POE_SELECTORS.threadRoots));
  let best = document.querySelector("main") || document.body;
  let maxCount = 0;

  for (const root of roots) {
    const count = Array.from(root.querySelectorAll(POE_SELECTORS.messageRows)).filter(poeVisible).length;
    if (count > maxCount) {
      maxCount = count;
      best = root;
    }
  }
  return best;
}

class PoeExtractor extends BaseExtractor {
  getPlatformName() {
    return "Poe";
  }

  extract() {
    const root = pickThreadRoot();
    const rows = Array.from(root.querySelectorAll(POE_SELECTORS.messageRows)).filter(poeVisible);
    const messages = [];
    const seen = new Set();

    for (const row of rows) {
      const role = getPoeRole(row);
      if (!role) continue;

      const contentNode = row.querySelector(POE_SELECTORS.messageContent) || row;
      const content = normalizePoeContent(cleanText(contentNode));
      if (!content) continue;

      const key = `${role}|${content}`;
      if (seen.has(key)) continue;
      seen.add(key);

      messages.push({ role, content, time: nowISO() });
    }

    const title = getPageTitle().replace(/\s*-\s*Poe$/i, "").trim() || "Untitled Chat";
    const model = cleanText(document.querySelector(POE_SELECTORS.model)) || null;

    return {
      title,
      model,
      url: getPageURL(),
      messages,
    };
  }
}

window.__aiArchiverExtractor = new PoeExtractor();
