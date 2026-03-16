---
description: Evaluate model for KC semantics
---

I'd like help understanding the semantic value of the KCs in the trained model. Could you use 'scripts/curate compare-sentences' (for example, scripts/curate contrast-sentences "食べる" "食べます" "食べない") to probe the model with a variety of Japanese sentences to see if you can tease out the meanings of some KCs. Please place your findings in ./semantics. Store per-KC findings in, for example, files like '.sematics/kc0000-<high level meaning>.txt'. Also, as you discover higher-level organizing principles--the way the model thinks about Japanese grammar, how KCs combine with each other, etc--please summarize organizing principles in semantics/organizing-principles.txt. I'm trying to understand the fitness of the model for the purpose of using individual KCs as SRS "learning points" that I could track for a user studying Japanese. Ideally, learning a particular KC would naturally lead the user to "unlock" new KCs such that they are kept at an efficient "learning frontier" (they are presented with sentences that they mostly understand but there are a small number of new concepts). In my mind, KCs should be binary-like in that they can be considered present or absent from a sentence based on the threshold the model chose for them. You're a Japanese language teaching expert, so please be thorough and explore the model based on your knowledge.

A few points:
- The model may mix grammar and vocabulary. This is expected, as the user must learn both when at an efficient learning frontier.
- The model may have discovered a model of understanding Japanese grammar that is different from the human way of teaching it. This may not be a disadvantage as the human way has trouble keeping the user at an efficient learning frontier.

Tools:
- You may now use curate find-kc-matches (ex ./scripts/curate find-kc-matches "+2,-1") to interrogate individual KCs for absence or presence
- scripts/curate contrast-sentences "食べる" "食べます" "食べない"