const SCRIPT_BY_HOST = [
  {
    host: "chatgpt.com",
    files: ["content_scripts/base_extractor.js", "content_scripts/chatgpt.js"],
  },
  {
    host: "claude.ai",
    files: ["content_scripts/base_extractor.js", "content_scripts/claude.js"],
  },
  {
    host: "gemini.google.com",
    files: ["content_scripts/base_extractor.js", "content_scripts/gemini.js"],
  },
  {
    host: "chat.deepseek.com",
    files: ["content_scripts/base_extractor.js", "content_scripts/deepseek.js"],
  },
  {
    host: "poe.com",
    files: ["content_scripts/base_extractor.js", "content_scripts/poe.js"],
  },
];

function getScriptFiles(url) {
  try {
    const host = new URL(url || "").hostname;
    const matched = SCRIPT_BY_HOST.find((item) => host.includes(item.host));
    return matched ? matched.files : null;
  } catch (error) {
    return null;
  }
}

async function getActiveTab() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0] || null;
}

async function sendExtract(tabId) {
  return chrome.tabs.sendMessage(tabId, { action: "extract" });
}

async function ensureInjected(tab) {
  const files = getScriptFiles(tab?.url || "");
  if (!files || tab?.id == null) return;
  await chrome.scripting.executeScript({ target: { tabId: tab.id }, files });
}

async function extractFromActiveTab() {
  const tab = await getActiveTab();
  if (!tab || tab.id == null) {
    return { success: false, error: "未找到活动标签页" };
  }

  try {
    const response = await sendExtract(tab.id);
    if (response?.success) return response;
    return { success: false, error: response?.error || "提取失败" };
  } catch (err) {
    const text = String(err?.message || err || "");
    if (!text.includes("Receiving end does not exist")) {
      return { success: false, error: text || "提取失败" };
    }
  }

  try {
    await ensureInjected(tab);
    const retry = await sendExtract(tab.id);
    if (retry?.success) return retry;
    return { success: false, error: retry?.error || "提取失败" };
  } catch (err) {
    return { success: false, error: String(err?.message || err || "提取失败") };
  }
}

chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
  if (request?.action === "extractActiveChat") {
    extractFromActiveTab()
      .then((resp) => sendResponse(resp))
      .catch((err) => sendResponse({ success: false, error: String(err?.message || err || "提取失败") }));
    return true;
  }

  if (request?.action === "openDashboard") {
    chrome.tabs.create({ url: "http://localhost:8765/dashboard" });
    sendResponse({ success: true });
    return true;
  }

  return false;
});
