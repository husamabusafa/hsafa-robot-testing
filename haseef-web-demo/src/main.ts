const apiKeyInput = document.getElementById("apiKey") as HTMLInputElement;
const haseefIdInput = document.getElementById("haseefId") as HTMLInputElement;
const skillInput = document.getElementById("skill") as HTMLInputElement;
const messageInput = document.getElementById("message") as HTMLTextAreaElement;
const sendBtn = document.getElementById("sendBtn") as HTMLButtonElement;
const statusDiv = document.getElementById("status") as HTMLDivElement;

const CORE_URL = "https://core.hsafa.com";

function showStatus(text: string, ok: boolean) {
  statusDiv.textContent = text;
  statusDiv.className = `status ${ok ? "ok" : "err"}`;
}

sendBtn.addEventListener("click", async () => {
  const apiKey = apiKeyInput.value.trim();
  const haseefId = haseefIdInput.value.trim();
  const skill = skillInput.value.trim() || "robot_base";
  const text = messageInput.value.trim();

  if (!apiKey) {
    showStatus("Please enter your HSAFA Core Key.", false);
    return;
  }
  if (!haseefId) {
    showStatus("Please enter a Haseef ID.", false);
    return;
  }
  if (!text) {
    showStatus("Please enter a message.", false);
    return;
  }

  sendBtn.disabled = true;
  statusDiv.className = "status";
  statusDiv.style.display = "none";

  try {
    const res = await fetch(`${CORE_URL}/api/v7/events`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
      },
      body: JSON.stringify({
        skill,
        type: "user_message",
        data: { text, source: "web-demo" },
        haseefId,
      }),
    });

    if (res.ok) {
      showStatus(`[OK] Event sent successfully.`, true);
      messageInput.value = "";
    } else {
      const body = await res.text();
      showStatus(`[FAIL] ${res.status} ${res.statusText}: ${body}`, false);
    }
  } catch (e) {
    showStatus(`[FAIL] ${(e as Error).message}`, false);
  } finally {
    sendBtn.disabled = false;
  }
});
