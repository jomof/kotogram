import json
import unittest
from unittest.mock import MagicMock, patch

from kotogram import cli


class TestCliCommands(unittest.TestCase):
    @patch("kotogram.cli._check_model")
    @patch("kotogram.analysis._ANALYZER.load")
    def test_vocab(self, mock_load, mock_check):
        mock_check.return_value = True
        mock_tokenizer = MagicMock()
        mock_tokenizer.get_vocab_sizes.return_value = {"f1": 100, "f2": 200}
        mock_load.return_value = (None, mock_tokenizer)

        with patch("builtins.print") as mock_print:
            args = MagicMock()
            cli.cmd_vocab(args)
            mock_print.assert_called()
            # check json output
            call_args = mock_print.call_args[0][0]
            data = json.loads(call_args)
            self.assertEqual(data, {"f1": 100, "f2": 200})

    @patch("kotogram.cli._check_model")
    @patch("kotogram.analysis._ANALYZER.load")
    def test_config(self, mock_load, mock_check):
        mock_check.return_value = True
        mock_model = MagicMock()
        mock_model.config.to_dict.return_value = {"d_model": 512}
        mock_load.return_value = (mock_model, None)

        with patch("builtins.print") as mock_print:
            args = MagicMock()
            cli.cmd_config(args)
            mock_print.assert_called()
            call_args = mock_print.call_args[0][0]
            data = json.loads(call_args)
            self.assertEqual(data, {"d_model": 512})

    def test_labels(self):
        with patch("builtins.print") as mock_print:
            args = MagicMock()
            cli.cmd_labels(args)
            mock_print.assert_called()
            call_args = mock_print.call_args[0][0]
            data = json.loads(call_args)
            self.assertIn("formality", data)
            self.assertIn("register", data)

    @patch("kotogram.cli._check_model")
    @patch("kotogram.cli._get_kotogram_from_args")
    @patch("kotogram.cli.grammar")
    def test_benchmark(self, mock_grammar, mock_get_koto, mock_check):
        mock_check.return_value = True
        mock_get_koto.return_value = "input_text"

        args = MagicMock()
        args.iterations = 5

        with patch("builtins.print") as mock_print:
            cli.cmd_benchmark(args)
            self.assertEqual(mock_grammar.call_count, 5)
            mock_print.assert_called()

    @patch("kotogram.cli._check_model")
    @patch("kotogram.augment.augment")
    def test_augment(self, mock_augment, mock_check):
        mock_check.return_value = True
        mock_augment.return_value = ["augmented"]

        args = MagicMock()
        args.text = "input"
        args.timeout = 1.0

        with patch("builtins.print") as mock_print:
            cli.cmd_augment(args)
            mock_augment.assert_called_with(["input"], timeout=1.0)
            mock_print.assert_called()
            call_args = mock_print.call_args[0][0]
            data = json.loads(call_args)
            self.assertEqual(data, ["augmented"])


if __name__ == "__main__":
    unittest.main()
