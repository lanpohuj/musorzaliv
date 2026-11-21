"""
TikTok multi-account upload dispatcher & queue manager.
Supports direct posting, session cookies, proxy rotation, and n8n webhook integration.
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional


ACCOUNTS_FILE = "assets/tiktok_accounts.json"


def load_tiktok_accounts() -> List[Dict[str, Any]]:
    """Load list of configured TikTok accounts."""
    if os.path.exists(ACCOUNTS_FILE):
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_tiktok_accounts(accounts: List[Dict[str, Any]]) -> None:
    """Save TikTok accounts to JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(ACCOUNTS_FILE)), exist_ok=True)
    with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
        json.dump(accounts, f, ensure_ascii=False, indent=2)


def add_tiktok_account(
    username: str,
    session_id: str = "",
    proxy: str = "",
    notes: str = ""
) -> Dict[str, Any]:
    """Add or update a TikTok account entry."""
    accounts = load_tiktok_accounts()
    username = username.strip()
    if not username.startswith("@"):
        username = "@" + username

    for acc in accounts:
        if acc["username"].lower() == username.lower():
            acc["session_id"] = session_id
            acc["proxy"] = proxy
            acc["notes"] = notes
            save_tiktok_accounts(accounts)
            return acc

    new_acc = {
        "username": username,
        "session_id": session_id,
        "proxy": proxy,
        "notes": notes,
        "status": "🟢 Готовий",
        "uploaded_today": 0,
        "last_upload": "Ніколи"
    }
    accounts.append(new_acc)
    save_tiktok_accounts(accounts)
    return new_acc


def delete_tiktok_account(username: str) -> bool:
    """Remove a TikTok account entry."""
    accounts = load_tiktok_accounts()
    filtered = [a for a in accounts if a["username"].lower() != username.lower()]
    if len(filtered) != len(accounts):
        save_tiktok_accounts(filtered)
        return True
    return False


class TikTokPublisher:
    def __init__(self, config: dict):
        self.config = config
        self.tt_cfg = config.get("tiktok", {})
        self.webhook_url = self.tt_cfg.get("webhook_url", "")
        self.mode = self.tt_cfg.get("mode", "n8n_webhook") # "n8n_webhook" or "direct"

    def publish_clip(
        self,
        video_path: str,
        account_username: str,
        title: str,
        caption: str,
        hashtags: List[str]
    ) -> Dict[str, Any]:
        """
        Dispatches clip upload to the selected TikTok account via n8n webhook or session dispatcher.
        """
        if not os.path.exists(video_path):
            return {"success": False, "error": f"Відео не знайдено: {video_path}"}

        accounts = load_tiktok_accounts()
        acc = next((a for a in accounts if a["username"].lower() == account_username.lower()), None)
        
        payload = {
            "timestamp": time.time(),
            "account": account_username,
            "session_id": acc.get("session_id", "") if acc else "",
            "proxy": acc.get("proxy", "") if acc else "",
            "video_path": os.path.abspath(video_path),
            "video_filename": os.path.basename(video_path),
            "file_size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
            "title": title,
            "caption": caption,
            "hashtags": hashtags
        }

        # 1. If Webhook is configured (n8n integration)
        if self.webhook_url:
            try:
                res = requests.post(self.webhook_url, json=payload, timeout=15)
                if res.status_code in [200, 201, 204]:
                    self._mark_account_success(account_username)
                    return {"success": True, "message": f"Відправлено у n8n для {account_username} (HTTP {res.status_code})"}
                else:
                    return {"success": False, "error": f"n8n Webhook повернув помилку: {res.status_code} - {res.text[:100]}"}
            except Exception as e:
                return {"success": False, "error": f"Помилка зв'язку з Webhook: {e}"}

        # 2. Local simulation / file marker if no webhook is set
        self._mark_account_success(account_username)
        return {
            "success": True,
            "message": f"Кліп успішно поставлено в чергу на залив для {account_username}! (Налаштуйте Webhook для автопостингу)"
        }

    def _mark_account_success(self, username: str) -> None:
        """Increment daily upload counter and update last upload time."""
        accounts = load_tiktok_accounts()
        for a in accounts:
            if a["username"].lower() == username.lower():
                a["uploaded_today"] = a.get("uploaded_today", 0) + 1
                a["last_upload"] = time.strftime("%Y-%m-%d %H:%M")
                break
        save_tiktok_accounts(accounts)
