const API_BASE = "http://127.0.0.1:8765";

const PLATFORM_MAP = {
  "chatgpt.com": "ChatGPT",
  "claude.ai": "Claude",
  "gemini.google.com": "Gemini",
  "chat.deepseek.com": "DeepSeek",
  "poe.com": "Poe",
};

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const platformBadge = document.getElementById("platformBadge");
const titleInput = document.getElementById("titleInput");
const tagsInput = document.getElementById("tagsInput");
const saveBtn = document.getElementById("saveBtn");
const messageBox = document.getElementById("messageBox");
const recentItems = document.getElementById("recentItems");

let currentPlatform = null;
let currentTabId = null;

function showMessage(text, type) {
  messageBox.textContent = text || "";
  messageBox.className = `msg ${type || ""}`;
}

function escapeHTML(text) {
  const div = document.createElement("div");
  div.textContent = text || "";
  return div.innerHTML;
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      statusDot.classList.add("online");
      statusText.textContent = "服务在线";
      return;
    }
  } catch (err) {
    // ignore
  }
  statusDot.classList.remove("online");
  statusText.textContent = "服务离线";
}

async function detectTabContext() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  const tab = tabs[0];
  if (!tab) return;

  currentTabId = tab.id;
  titleInput.value = tab.title || "";

  try {
    const url = new URL(tab.url || "");
    const host = url.hostname;
    currentPlatform = null;
    for (const [domain, platform] of Object.entries(PLATFORM_MAP)) {
      if (host.includes(domain)) {
        currentPlatform = platform;
        break;
      }
    }
  } catch (err) {
    currentPlatform = null;
  }

  platformBadge.textContent = `当前平台：${currentPlatform || "未识别"}`;
}

async function requestExtractFromActiveTab() {
  const response = await chrome.runtime.sendMessage({ action: "extractActiveChat" });
  if (!response?.success) {
    throw new Error(response?.error || "提取失败");
  }
  return response.data;
}

async function saveCurrentChat() {
  if (!currentTabId) {
    showMessage("未找到活动标签页", "err");
    return;
  }
  if (!currentPlatform) {
    showMessage("当前页面不在支持的平台列表", "err");
    return;
  }

  saveBtn.disabled = true;
  showMessage("正在提取并保存...", "");

  try {
    const extracted = await requestExtractFromActiveTab();
    const messages = extracted?.messages || [];
    if (messages.length === 0) {
      throw new Error("未检测到对话内容，请确保页面上有对话消息");
    }

    const payload = {
      platform: currentPlatform,
      model: extracted.model || null,
      title: (titleInput.value || extracted.title || "Untitled Chat").trim(),
      url: extracted.url || null,
      tags: (tagsInput.value || "")
        .split(",")
        .map((t) => t.trim())
        .filter((t) => t.length > 0),
      messages,
    };

    const res = await fetch(`${API_BASE}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      let detail = "保存失败";
      try {
        const errData = await res.json();
        detail = errData?.detail || detail;
      } catch (err) {
        // ignore
      }
      throw new Error(detail);
    }

    const data = await res.json();
    showMessage(`保存成功：${data.id}`, "ok");
    await loadRecent();
  } catch (error) {
    showMessage(`保存失败：${error.message || "未知错误"}`, "err");
  } finally {
    saveBtn.disabled = false;
  }
}

async function loadRecent() {
  recentItems.innerHTML = "<div class='item'>加载中...</div>";
  try {
    const res = await fetch(`${API_BASE}/chats?limit=5`);
    if (!res.ok) throw new Error("加载失败");

    const data = await res.json();
    const chats = data?.chats || [];
    if (chats.length === 0) {
      recentItems.innerHTML = "<div class='item'>暂无记录</div>";
      return;
    }

    recentItems.innerHTML = chats
      .map(
        (chat) => `
          <div class="item">
            <div class="t" title="${escapeHTML(chat.title)}">${escapeHTML(chat.title)}</div>
            <div class="p">${escapeHTML(chat.platform)} · ${escapeHTML(chat.saved_at || "")}</div>
          </div>
        `
      )
      .join("");
  } catch (error) {
    recentItems.innerHTML = "<div class='item'>无法连接后端</div>";
  }
}

saveBtn.addEventListener("click", saveCurrentChat);

(async function init() {
  await Promise.all([checkHealth(), detectTabContext(), loadRecent()]);
})();
