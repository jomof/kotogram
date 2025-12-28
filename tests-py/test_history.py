import os
import shutil
import tempfile
import unittest

from train import history


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.history_path = os.path.join(self.test_dir, "history.tsv")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_append_read_epoch_event(self):
        event = history.StyleEpochEvent(1, {"loss": 0.5})
        history.append_event(self.history_path, event)

        events = history.read_events(self.history_path)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], history.StyleEpochEvent)
        self.assertEqual(events[0].epoch, 1)
        self.assertEqual(events[0].metrics["loss"], 0.5)

    def test_append_multiple(self):
        e1 = history.KcEpochEvent(1, {"acc": 0.9})
        e2 = history.StyleEpochEvent(1, {"acc": 0.8})

        history.append_event(self.history_path, e1)
        history.append_event(self.history_path, e2)

        events = history.read_events(self.history_path)
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], history.KcEpochEvent)
        self.assertEqual(events[0].metrics["acc"], 0.9)
        self.assertIsInstance(events[1], history.StyleEpochEvent)

    def test_clear(self):
        e1 = history.KcEpochEvent(1, {})
        history.append_event(self.history_path, e1)
        history.clear_history(self.history_path)
        self.assertFalse(os.path.exists(self.history_path))
        events = history.read_events(self.history_path)
        self.assertEqual(events, [])

    def test_read_missing(self):
        events = history.read_events(os.path.join(self.test_dir, "missing.tsv"))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
