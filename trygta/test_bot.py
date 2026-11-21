import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import bot


class EditorialFlow(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.enterContext(patch.object(bot, "DB", folder.name + "/queue"))
        self.enterContext(patch.dict(os.environ, OWNER_ID="123"))
        bot.init()
        self.packet = {"data":[{"id":"456", "text":"GTA VI hidden map detail in a trailer",
                       "created_at":datetime.now(timezone.utc).isoformat(),"author_id":"1",
                       "public_metrics":{"like_count":150},"attachments":{"media_keys":["m"]}}],
                       "includes":{"users":[{"id":"1","username":"tester"}],
                                   "media":[{"type":"photo","media_key":"m","url":"https://pbs.twimg.com/example"}]}}
        bot.ingest(self.packet)
        bot.ingest(self.packet)
        self.api = self.enterContext(patch.object(bot,"tg",return_value={"message_id":22}))
        self.enterContext(patch.object(bot,"render",return_value="<b>Деталь карты</b>"))
        bot.preview()

    def press(self, action, user=123, **message_fields):
        bot.callback({"id":"query","from":{"id":user},"message":{"message_id":22,"chat":{"id":123},**message_fields},
                      "data":action+".456"})

    def test_approval(self):
        self.assertEqual(self.api.call_args.args,("sendPhoto",))
        self.assertTrue(self.api.call_args.kwargs["show_caption_above_media"])
        self.press("yes",987)
        self.assertEqual(bot.rows("SELECT state FROM posts")[0][0],"review")
        self.press("yes",caption="Деталь карты",show_caption_above_media=True,
                   caption_entities=[{"type":"bold","offset":0,"length":12}])
        self.press("yes")
        copies = [c for c in self.api.call_args_list if c.args==("copyMessage",)]
        self.assertEqual(len(copies),1)
        self.assertEqual(copies[0].kwargs["chat_id"],"@trygta")
        self.assertTrue(copies[0].kwargs["show_caption_above_media"])
        self.assertEqual(copies[0].kwargs["caption"],"Деталь карты")
        self.assertEqual(len(bot.rows("SELECT * FROM posts")),1)

    def test_reject(self):
        self.press("no")
        self.press("yes")
        self.assertEqual(bot.rows("SELECT state FROM posts")[0][0],"rejected")
        self.assertFalse(any(c.args==("copyMessage",) for c in self.api.call_args_list))

    def test_timeout(self):
        self.api.side_effect = TimeoutError
        with self.assertRaises(TimeoutError):
            self.press("yes")
        bot.init()
        self.assertEqual(bot.rows("SELECT state FROM posts")[0][0],"uncertain")

    def test_edit_photo_by_reply(self):
        self.api.return_value = {"message_id":31}
        self.press("edit",photo=[{"file_id":"example"}])
        bot.apply_edit({"chat":{"id":123},"reply_to_message":{"message_id":30},"text":"Не тот ответ"})
        bot.apply_edit({"chat":{"id":123},"reply_to_message":{"message_id":31},"text":"Новый <заголовок>\n\nНовый текст."})
        edits = [c for c in self.api.call_args_list if c.args==("editMessageCaption",)]
        self.assertEqual(len(edits),1)
        self.assertEqual(edits[0].kwargs["message_id"],22)
        self.assertTrue(edits[0].kwargs["show_caption_above_media"])
        draft = bot.rows("SELECT draft FROM posts")[0][0]
        self.assertIn("<b>Новый &lt;заголовок&gt;</b>", draft)
        self.assertIn("@tester", draft)
        self.assertEqual(bot.rows("SELECT state FROM posts")[0][0],"review")
        self.assertFalse(any(c.args==("copyMessage",) for c in self.api.call_args_list))
