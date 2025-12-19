
import sys
import os
import shutil
import tempfile
import unittest
import csv
# Add project root to path so we can import scripts and kotogram
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.label import main as label_main, compute_labels
from scripts.cache import get_kotogram_cache, ShardedKotogramCache
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.analysis import RegisterLevel

class TestLabelScript(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.shards_dir = os.path.join(self.test_dir, "shards")
        self.data_file = os.path.join(self.test_dir, "test_data.tsv")
        
        # Create dummy data
        with open(self.data_file, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["id1", "label1", "これはテストです。"])
            writer.writerow(["id2", "label2", "美味しいですね。"])
            writer.writerow(["id3", "label3", "走る。"])

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # Reset global cache instance if needed
        import scripts.cache
        scripts.cache._kotogram_cache = None

    def test_label_and_cache(self):
        # Run label.py via main
        import sys
        from unittest.mock import patch
        
        # Mock sys.argv
        test_args = [
            "scripts/label.py", 
            "--data", self.data_file, 
            "--num-workers", "1"
        ]
        
        # Point the cache to our temp dir
        with patch('scripts.cache.ShardedKotogramCache.DEFAULT_SHARDS_DIR', self.shards_dir):
             # Ensure the global cache is reset to use the new DEFAULT_SHARDS_DIR
             import scripts.cache
             scripts.cache._kotogram_cache = None
             
             with patch.object(sys, 'argv', test_args):
                 label_main()
        
        # Verify cache was created and populated
        # Re-initialize to see what happened
        import scripts.cache
        scripts.cache._kotogram_cache = None
        cache = get_kotogram_cache(self.shards_dir)
        
        print(f"Checking results in {self.shards_dir}...")
        results = cache.get_batch(["これはテストです。", "美味しいですね。", "走る。"])
        
        for k_sent, v in results.items():
            if v is None:
                print(f"MISSING: {k_sent}")
        
        self.assertIsNotNone(results["これはテストです。"])
        self.assertIsNotNone(results["美味しいですね。"])
        self.assertIsNotNone(results["走る。"])
        
        # Check if fields are populated
        k, f, g_val, g_prag, r_lbls = results["これはテストです。"]
        self.assertTrue(len(k) > 0)
        self.assertIsNotNone(f)
        self.assertIsNotNone(g_val)
        self.assertIsNotNone(g_prag)
        self.assertIsNotNone(r_lbls)

    def test_incremental_labeling(self):
         # First run
        import sys
        from unittest.mock import patch
        
        with patch('scripts.cache.ShardedKotogramCache.DEFAULT_SHARDS_DIR', self.shards_dir):
            import scripts.cache
            scripts.cache._kotogram_cache = None
            
            test_args = ["scripts/label.py", "--data", self.data_file, "--num-workers", "1"]
            with patch.object(sys, 'argv', test_args):
                 label_main()
            
            # Verify something was written
            files = os.listdir(self.shards_dir)
            print(f"Shard files: {files}")
            if not files:
                self.fail("No shard files created")
                
            shard_path = os.path.join(self.shards_dir, files[0])
            mtime1 = os.path.getmtime(shard_path)
            
            # Second run with same data
            with patch.object(sys, 'argv', test_args):
                 label_main()
            
            # Add a new file
            new_data_file = os.path.join(self.test_dir, "new_data.tsv")
            with open(new_data_file, 'w', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["id4", "label4", "新しい文です。"])
            
            test_args_new = ["scripts/label.py", "--data", self.data_file, "--agrammatic-sentences", new_data_file, "--num-workers", "1"]
            with patch.object(sys, 'argv', test_args_new):
                 label_main()
            
            scripts.cache._kotogram_cache = None
            cache = get_kotogram_cache(self.shards_dir)
            results = cache.get_batch(["新しい文です。"])
            self.assertIsNotNone(results["新しい文です。"])


class TestLabeling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize parser once as it is heavy
        cls.parser = SudachiJapaneseParser()

    def test_mixed_registers_real_data(self):
        # Real data from jpn_agrammatic_unpragmatic.tsv
        # 1297_unprag_g3	jpn	きみにちょっとしたものをもってきたよぜわ。
        # "ぜわ" mixes masculine "ze" and feminine "wa"
        sentence = "きみにちょっとしたものをもってきたよぜわ。"
        kotogram = self.parser.japanese_to_kotogram(sentence)
        
        result = compute_labels(kotogram, sentence)
        
        # Verify it is Unpragmatic (0) and Value is 0.0
        self.assertEqual(result.gender_pragmatic, 0, "Mixed registers 'ぜわ' should be Unpragmatic (0)")
        self.assertEqual(result.gender_value, 0.0, "Mixed registers should have neutral value (0.0)")
        
        self.assertTrue(len(result.register_ids) >= 2, "Should detect multiple registers for 'ぜわ'")

    def test_neutral_sentence(self):
        sentence = "これはペンです。"
        kotogram = self.parser.japanese_to_kotogram(sentence)
        result = compute_labels(kotogram, sentence)
        
        self.assertEqual(result.gender_pragmatic, 1, "Neutral sentence should be Pragmatic (1)")
        self.assertEqual(result.gender_value, 0.0, "Neutral sentence should have value 0.0")

    def test_masculine_sentence(self):
        sentence = "行くぞ。" # "ぞ" is masculine
        kotogram = self.parser.japanese_to_kotogram(sentence)
        result = compute_labels(kotogram, sentence)
        
        self.assertEqual(result.gender_pragmatic, 1, "Masculine sentence should be Pragmatic (1)")
        self.assertEqual(result.gender_value, -1.0, "Masculine sentence should have value -1.0")

    def test_feminine_sentence(self):
        sentence = "行くわ。" # "わ" (rising) is feminine
        kotogram = self.parser.japanese_to_kotogram(sentence)
        result = compute_labels(kotogram, sentence)
        
        self.assertEqual(result.gender_pragmatic, 1, "Feminine sentence should be Pragmatic (1)")
        self.assertEqual(result.gender_value, 1.0, "Feminine sentence should have value 1.0")

    def test_register_override(self):
        sentence = "overridden_sentence"
        kotogram = "mock_kotogram"
        
        # Manually pass overrides
        overrides = [RegisterLevel.BUSHI]
        result = compute_labels(kotogram, sentence, register_overrides=overrides)
        
        # BUSHI implies masculine if neutral
        self.assertEqual(result.gender_value, -1.0, "BUSHI override should imply Masculine")


if __name__ == '__main__':
    unittest.main()
