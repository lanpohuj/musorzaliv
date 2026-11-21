"""Budgeted Apify polling with a persistent run handle and shared rolling allowance."""
import hashlib
import json
import os
import re
import time
from contextlib import closing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError

API = "https://api.apify.com/v2/"
ACTOR = "wAusCMrm284Voaw86"
QUERY = '("GTA VI" OR "GTA 6" OR GTA6 OR GTAVI OR "Grand Theft Auto VI") lang:en -filter:retweets -filter:replies min_faves:10'
TOPICS = "(secret OR location OR glitch OR bug OR mod OR mods OR hidden OR \"easter egg\" OR feature OR mission OR mechanic OR gameplay OR update OR trailer)"
BASE_INTERVAL = 1800
WINDOW = 31 * 86400
RESERVE = 0.003
PER_ACCOUNT = 4.95
MAX_ACCOUNTS = 10
ITEMS = 10
TERMINAL = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}


def credentials():
    return list(dict.fromkeys(re.findall(r'apify_api_[A-Za-z0-9_-]+', os.getenv("APIFY_API_TOKENS", ""))))


def key_id(token):
    return hashlib.sha256(token.encode()).hexdigest()


def packet(row):
    ident = str(row.get("id", ""))
    body = row.get("text")
    if not ident.isdigit() or not body or row.get("lang") != "en" or row.get("isReply") or row.get("isRetweet"):
        return None
    stamp = row.get("createdAt") or ""
    try:
        dt = parsedate_to_datetime(stamp)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(stamp.replace("Z","+00:00"))
        except ValueError:
            return None
    author = row.get("author") or {}
    photo = next((m.get("mediaUrl") or m.get("url") for m in row.get("media") or []
                  if isinstance(m,dict) and m.get("type")=="photo"), None)
    metrics = {k+"_count":int(row.get(k+"Count") or 0) for k in ("like","reply","retweet","quote")}
    return {"data":{"id":ident,"text":body,"created_at":dt.isoformat(),"author_id":"u",
                    "public_metrics":metrics,"attachments":{"media_keys":["p"] if photo else []}},
            "includes":{"users":[{"id":"u","username":author.get("username") or ""}],
                        "media":[{"media_key":"p","type":"photo","url":photo}] if photo else []}}


class Feed:
    def __init__(self, bot):
        self.bot = bot
        bot.change("""CREATE TABLE IF NOT EXISTS apify_runs
                   (started REAL PRIMARY KEY, token TEXT, run TEXT, state TEXT, cost REAL)""")
        bot.change("CREATE TABLE IF NOT EXISTS apify_keys(token TEXT PRIMARY KEY, until REAL)")
        self.count = min(MAX_ACCOUNTS,len(credentials()))
        self.interval = BASE_INTERVAL / max(self.count,1)
        self.limit = PER_ACCOUNT * self.count
        self.status = "подключение"

    def api(self, token, path, **kwargs):
        return self.bot.call(API+path,bearer=token,**kwargs)

    def reserve(self, token, now):
        with closing(self.bot.db()) as c, c:
            c.execute("BEGIN IMMEDIATE")
            last,total = c.execute("SELECT max(started),coalesce(sum(cost),0) FROM apify_runs WHERE started>?",(now-WINDOW,)).fetchone()
            if (last and now-last<self.interval) or total+RESERVE>self.limit:
                return False
            c.execute("INSERT INTO apify_runs VALUES(?,?,NULL,'reserved',?)",(now,key_id(token),RESERVE))
            return True

    def run(self):
        b = self.bot
        tokens = credentials()
        active = b.rows("SELECT * FROM apify_runs WHERE state='running' ORDER BY started LIMIT 1")
        if active:
            r = active[0]
            token = next((v for v in tokens if key_id(v)==r["token"]), None)
            if not token:
                self.status = "нужен исходный токен для чтения запущенного задания"
                return
            info = self.api(token,"actor-runs/"+r["run"])["data"]
            if info["status"] not in TERMINAL:
                return
            count = 0
            if info["status"]=="SUCCEEDED":
                items = self.api(token,"datasets/"+info["defaultDatasetId"]+"/items",params={"clean":"true","limit":ITEMS})
                for row in items:
                    item = packet(row)
                    if item:
                        b.ingest(item)
                        count += 1
            reported = info.get("usageTotalUsd")
            cost = RESERVE if reported is None else float(reported) + 0.00005
            summary = f'{info["status"]}, получено {count}, {datetime.now(timezone.utc):%H:%M} UTC'
            b.change("UPDATE apify_runs SET state=?,cost=? WHERE started=?",(summary,cost,r["started"]))
            self.status = summary
            return
        now = time.time()
        last = b.rows("SELECT started,state FROM apify_runs ORDER BY started DESC LIMIT 1")
        if last and now-last[0]["started"]<self.interval:
            self.status = last[0]["state"]
            return
        total = b.rows("SELECT coalesce(sum(cost),0) FROM apify_runs WHERE started>?",(now-WINDOW,))[0][0]
        if total+RESERVE>self.limit:
            self.status = "общий лимит на 31 день исчерпан"
            return
        for token in tokens[:MAX_ACCOUNTS]:
            ident = key_id(token)
            if b.rows("SELECT 1 FROM apify_keys WHERE token=? AND until>?",(ident,now)):
                continue
            try:
                plan = self.api(token,"users/me")["data"]["plan"]
                usage = self.api(token,"users/me/usage/monthly")["data"]
                free = plan["id"]=="FREE" and 0 < plan["maxMonthlyUsageUsd"] <= plan["monthlyUsageCreditsUsd"]
                if not free or float(usage["totalUsageCreditsUsdAfterVolumeDiscount"])+0.05>min(5,plan["monthlyUsageCreditsUsd"]):
                    end = datetime.fromisoformat(usage["usageCycle"]["endAt"].replace("Z","+00:00")).timestamp()
                    b.change("INSERT OR REPLACE INTO apify_keys VALUES(?,?)",(ident,end+1))
                    continue
                cycle_start = datetime.fromisoformat(usage["usageCycle"]["startAt"].replace("Z","+00:00")).timestamp()
                booked = b.rows("SELECT coalesce(sum(cost),0) FROM apify_runs WHERE token=? AND started>=?",(ident,cycle_start))[0][0]
                if booked + RESERVE > PER_ACCOUNT:
                    end = datetime.fromisoformat(usage["usageCycle"]["endAt"].replace("Z","+00:00")).timestamp()
                    b.change("INSERT OR REPLACE INTO apify_keys VALUES(?,?)",(ident,end+1))
                    continue
                if not self.reserve(token,now):
                    return
                launches = b.rows("SELECT count(*) FROM apify_runs")[0][0]
                query = QUERY if launches % 5 == 0 else QUERY+" "+TOPICS
                info = self.api(token,"acts/"+ACTOR+"/runs",
                    params={"memory":256,"timeout":60,"maxItems":ITEMS,"maxTotalChargeUsd":"0.0015"},
                    payload={"twitterContent":query,"queryType":"Top" if launches % 3 == 0 else "Latest",
                             "since_time":str(int(now)-43200),"maxItems":ITEMS,
                             "inputStatus":False})["data"]
                b.change("UPDATE apify_runs SET run=?,state='running' WHERE started=?",(info["id"],now))
                self.status = "идёт поиск"
                return
            except HTTPError as e:
                if e.code not in (401,402,403,429):
                    raise
                b.change("INSERT OR REPLACE INTO apify_keys VALUES(?,?)",(ident,now+(900 if e.code==429 else 21600)))
                b.change("DELETE FROM apify_runs WHERE started=? AND token=? AND state='reserved'",(now,ident))
        self.status = "нет доступного токена бесплатного тарифа"
