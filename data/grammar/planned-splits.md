# Planned GP Splits

Candidates identified by intra-GP KC clustering (`curate find-nuance-divisions`).
Only splits reflecting genuine semantic/functional differences are listed here.

**General finding**: 1,336 of 1,370 GPs show *some* sub-clustering, but the dominant
split dimension is **register** (casual だ vs polite です vs formal である). This
confirms the model encodes register as a primary KC dimension but does not indicate
missing nuance divisions for most GPs.

## Genuine Semantic/Functional Splits

に (agent・source)|Split into "に (passive agent)" — marks the performer in passive constructions (先生に褒められた, 犬に噛まれた); and "に (source・giver)" — marks the origin in giving/receiving/learning (友達に本を借りた, 父にお金をもらった). k=4 sub-clusters cleanly separate these two functions (×2 for register), with nearest GPs Verb[(ら)れる] (direct passive) and Verb[(ら)れる] (adversative passive) for the agent clusters. These are distinct grammatical roles taught separately in pedagogy.

ところを (while or in circumstances)|Split into "ところを (narrative timing)" — marks the moment something is interrupted (出かけようとしているところを、急な来客があった); and "ところを (apologetic formula)" — fixed polite expression acknowledging interruption (お忙しいところを申し訳ありません). Sub-cluster 1 aligns with ところに・ところへ (sim=0.885); sub-cluster 2 is a distinct pragmatic routine with no close GP match.

Verb[て] (casual request)|Split into "Verb[て・てくれ] (direct request)" — bare て-form or てくれ as a straightforward command (あっちに行って, おい、それ貸してくれ); and "Verb[てよ・てね] (softened request)" — て-form plus sentence-final particles that change the pragmatic force: てよ adds emotional emphasis or complaint (誰か助けてよ！), てね adds gentle guidance (こちらに座ってね). Different speech act forces, different social contexts.

## Register-Only Splits (not actionable, but informative)

The following top-scoring GPs split cleanly on register alone. No new GPs needed, but
confirms the model treats casual/polite/formal as distinct KC patterns:

- てくれてありがとう: ありがとうございます vs ありがとう
- 唯一: だ / である / です (3-way register split)
- お〜ください: ください vs くださいませ
- [Verb-stem]にくい: plain vs です-form
- い-Adjective[くない]: plain vs です-form
- い-Adjective[くなかった]: plain / casual-particle / ありませんでした (3-way)
- でなくてなんだろう: literary vs casual
- Verb[てしまう・ちゃう・じゃう]: casual vs polite
- て仕方がない: casual vs polite (also shows しょうがない ↔ 仕方がない lexical variants)
