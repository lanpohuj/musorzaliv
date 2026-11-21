import io
import os
import unittest
from email.message import Message
from unittest.mock import patch
from urllib.error import HTTPError

import bot


class PhotoUpload(unittest.TestCase):
    def test_retry_with_original_bytes(self):
        error = HTTPError("https://example.invalid",400,"bad request",{},io.BytesIO(
            b'{"description":"Bad Request: failed to get HTTP URL content"}'))
        original = io.BytesIO(b"original-photo-bytes")
        original.headers = Message()
        original.headers["Content-Type"] = "image/jpeg"
        with patch.object(bot,"tg",side_effect=error), patch.dict(os.environ,BOT_TOKEN="test"), \
                patch.object(bot,"urlopen",side_effect=[original,io.BytesIO(b'{"result":{"message_id":42}}')]) as opens:
            result = bot.send_photo("https://pbs.twimg.com/media/example.jpg","Caption",chat_id="123",
                                    reply_markup={"inline_keyboard":[]})
        self.assertEqual(result["message_id"],42)
        request = opens.call_args.args[0]
        for expected in (b"original-photo-bytes",b'name="photo"',b"Caption",b"inline_keyboard"):
            self.assertIn(expected,request.data)

    def test_uncertain_delivery_not_resent(self):
        with patch.object(bot,"tg",side_effect=TimeoutError), patch.object(bot,"urlopen") as upload:
            with self.assertRaises(TimeoutError):
                bot.send_photo("https://pbs.twimg.com/media/example.jpg","Caption",chat_id="123")
            upload.assert_not_called()
