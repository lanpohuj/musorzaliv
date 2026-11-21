import os
import tempfile
import unittest
from unittest.mock import patch
import bot
from apify_feed import Feed, RESERVE, packet


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(bot,"DB",self.folder+"/db"))
        self.enterContext(patch.dict(os.environ,APIFY_API_TOKENS="apify_api_first,apify_api_second"))
        bot.init()
        self.feed = Feed(bot)

    def test_rotation(self):
        def mock(token,path,**kwargs):
            if path=="users/me":
                return {"data":{"plan":{"id":"FREE","maxMonthlyUsageUsd":5,"monthlyUsageCreditsUsd":5}}}
            if path.endswith("/monthly"):
                return {"data":{"totalUsageCreditsUsdAfterVolumeDiscount":5 if token.endswith("first") else 0,
                                "usageCycle":{"startAt":"2025-01-01T00:00:00Z","endAt":"2099-01-01T00:00:00Z"}}}
            self.assertEqual(token,"apify_api_second")
            self.assertEqual(kwargs["params"]["maxTotalChargeUsd"],"0.0015")
            return {"data":{"id":"test-run"}}
        with patch.object(self.feed,"api",side_effect=mock):
            self.feed.run()
        self.assertEqual(bot.rows("SELECT run FROM apify_runs")[0][0],"test-run")
        self.assertEqual(len(bot.rows("SELECT * FROM apify_keys")),1)

    def test_rolling_reservation(self):
        self.assertTrue(self.feed.reserve("a",100000))
        self.assertFalse(self.feed.reserve("b",100001))
        self.feed.limit = RESERVE
        self.assertFalse(self.feed.reserve("b",105000))

    def test_photo_mapping(self):
        item=packet({"id":"123","text":"New GTA VI location","lang":"en","createdAt":"Thu Nov 19 13:34:15 +0000 2026",
                     "author":{"username":"sample"},"media":[{"type":"photo","mediaUrl":"https://pbs.twimg.com/example"}],
                     "likeCount":42})
        self.assertEqual(item["includes"]["media"][0]["url"],"https://pbs.twimg.com/example")
        self.assertEqual(item["data"]["public_metrics"]["like_count"],42)
