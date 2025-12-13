
from kotogram.sudachi_japanese_parser import SudachiJapaneseParser
from kotogram.analysis import grammaticality, style

parser = SudachiJapaneseParser(dict_type='full')

sentences = [
    "駐車場がありませんホテルは不便です。",
    "駐車 場 が あり ませ ん ホテル は 不便 です。",  # Space separated (as in script)
    "新しいスマホ、できたら今すぐ買いたいだ。",
    "新 しい スマホ 、 でき たら 今 すぐ 買い たい だ 。", # Space separated
    "冷たい水があります自動販売機。",
    "英語のHelloは、日本語のこんにちはに当たる。"  # Control (Good)
]

print("Checking grammaticality...")
for sent in sentences:
    kotogram = parser.japanese_to_kotogram(sent)
    is_grammatic = grammaticality(kotogram, use_model=True)
    # Also get full style info
    s = style(kotogram, use_model=True)
    
    print(f"\nSentence: {sent}")
    print(f"Grammatical: {is_grammatic}")
    print(f"Style Full: {s}")
    print(f"Kotogram: {kotogram}")

