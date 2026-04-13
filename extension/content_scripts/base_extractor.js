class BaseExtractor {
  getPlatformName() {
    throw new Error("Not implemented");
  }

  extract() {
    throw new Error("Not implemented");
  }
}

function cleanText(el) {
  if (!el) return "";
  return (el.innerText || el.textContent || "")
    .replace(/\u00a0/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function nowISO() {
  return new Date().toISOString();
}

function getPageTitle() {
  return (document.title || "").trim();
}

function getPageURL() {
  return window.location.href;
}

function uniqueMessages(messages) {
  const out = [];
  const seen = new Set();
  for (const msg of messages || []) {
    const role = (msg.role || "").toLowerCase();
    const content = (msg.content || "").trim();
    if (!content || (role !== "user" && role !== "assistant" && role !== "system")) {
      continue;
    }
    const key = `${role}|${content}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ role, content, time: msg.time || nowISO() });
  }
  return out;
}

if (!window.__aiArchiverListenerInstalled) {
  chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
    if (request?.action !== "extract") return;

    try {
      const extractor = window.__aiArchiverExtractor;
      if (!extractor || typeof extractor.extract !== "function") {
        sendResponse({ success: false, error: "未找到页面提取器" });
        return;
      }
      const data = extractor.extract();
      data.messages = uniqueMessages(data.messages);
      sendResponse({ success: true, data });
    } catch (error) {
      sendResponse({ success: false, error: error?.message || "提取失败" });
    }
  });
  window.__aiArchiverListenerInstalled = true;
}

window.BaseExtractor = BaseExtractor;
window.cleanText = cleanText;
window.nowISO = nowISO;
window.getPageTitle = getPageTitle;
window.getPageURL = getPageURL;
window.uniqueMessages = uniqueMessages;
