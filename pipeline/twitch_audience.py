"""Optional VOD discovery from publicly available, viewer-made Twitch clips."""
import logging

import httpx
from yt_dlp import YoutubeDL
from yt_dlp.extractor.twitch import TwitchBaseIE


def seeds(edges, vod, duration, limit=10):
    result = []
    for clip in sorted((e["node"] for e in edges), key=lambda c: c.get("viewCount") or 0, reverse=True):
        if (clip.get("video") or {}).get("id") != vod:
            continue
        offset = clip.get("videoOffsetSeconds")
        length = clip.get("durationSeconds")
        if not isinstance(offset, (int, float)) or not length or offset <= 0:
            continue
        # Twitch's web GraphQL offset marks the END of a viewer-created cut.
        start, end = max(0, offset-length), min(duration, offset)
        if end-start < 3:
            continue
        if any(min(end,s["end"])-max(start,s["start"]) > .5*min(end-start,s["end"]-s["start"]) for s in result):
            continue
        result.append(dict(start=start, peak=(start+end)/2, end=end, kind="reaction",
                           title=clip["title"], audience_views=clip["viewCount"],
                           evidence="Viewer-marked interval. The midpoint is only a search hint. "
                                    "Find the actual opening and payoff in the supplied footage."))
        if len(result) >= limit:
            break
    return result


def viewer_moments(vod, duration):
    if not vod:
        return []
    fields = "edges{node{title viewCount videoOffsetSeconds durationSeconds video{id}}}"
    edges = []
    try:
        with YoutubeDL({"quiet": True}) as dl, httpx.Client(timeout=8, headers={
                "Client-ID": TwitchBaseIE(dl)._CLIENT_ID}) as client:
            def query(document, variables):
                response = client.post("https://gql.twitch.tv/gql", json={"query": document, "variables": variables})
                response.raise_for_status()
                return response.json()["data"]
            info = query("query($id:ID!){video(id:$id){owner{login} clips(first:100){" + fields + "}}}", {"id": vod})
            video = info["video"]
            edges = video["clips"]["edges"]
            for period in ["LAST_WEEK", "LAST_MONTH", "ALL_TIME"]:
                data = query("query($login:String!){user(login:$login){clips(first:100,criteria:{period:" +
                             period + "}){" + fields + "}}}", {"login": video["owner"]["login"]})
                ranked = data["user"]["clips"]["edges"]
                edges.extend(ranked)
                if len(seeds(ranked, vod, duration)) >= 10:
                    break
    except (httpx.HTTPError, KeyError, TypeError, ValueError, AttributeError):
        logging.getLogger(__name__).info("Twitch viewer clips unavailable for VOD %s", vod)
    return seeds(edges, vod, duration)
