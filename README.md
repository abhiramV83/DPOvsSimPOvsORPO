# English–Telugu Preference Optimization: DPO vs SimPO vs ORPO on a Self-Built Dataset

A controlled comparison of three preference-optimization methods — **DPO**, **SimPO**, and **ORPO** — fine-tuned on top of [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) for English→Telugu translation, using a preference dataset built from scratch and evaluated with an LLM-judge tournament, cross-checked by a native Telugu speaker.

## Why this project

Most public preference-optimization demos are English-to-English (chat, summarization, code). Regional/low-resource languages are almost entirely absent from this literature, despite alignment techniques mattering just as much — arguably more — for languages where base models are weaker to begin with. This project asks a simple question: **on a small, self-built Telugu preference dataset, does preference optimization actually beat plain supervised fine-tuning?**

## Headline result

| Model | Tournament wins (out of 360 matchups) |
|---|---|
| **SFT (baseline)** | **134** |
| DPO | 106 |
| SimPO | 65 |
| ORPO | 55 |

**Plain SFT won the most head-to-head matchups — outperforming all three preference-optimization methods.** This is the central finding of the project, and it runs counter to the default assumption that preference optimization should improve on an SFT baseline. See [Discussion](#discussion--why-sft-won) for analysis.

## Methods compared

All three methods were trained on top of a shared **SFT checkpoint** (except ORPO — see note), using LoRA/QLoRA adapters on the same base model, so the only variable across runs is the alignment loss itself.

- **SFT** — supervised fine-tuning on the `chosen` translations only. No preference signal. This is both the baseline and, for DPO/SimPO, the starting checkpoint.
- **DPO** ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) — a sigmoid loss over the log-probability ratio of `chosen` vs `rejected`, measured relative to a frozen reference copy of the SFT model.
- **SimPO** ([Meng et al., 2024](https://arxiv.org/abs/2405.14734)) — reference-free; uses the model's own length-normalized log-likelihood as the implicit reward, plus a fixed target margin. Lower memory footprint than DPO since no second model is held in memory.
- **ORPO** ([Hong et al., 2024](https://arxiv.org/abs/2403.07691)) — combines imitation learning and preference alignment into a single training stage via an odds-ratio penalty. Trained from the **raw base model**, not the SFT checkpoint, since it doesn't require a separate SFT stage.

## Dataset

Built entirely from scratch — no existing Telugu preference dataset was available at this scale.

1. **Source sentences:** 400 English sentences sampled from [Samanantar](https://huggingface.co/datasets/ai4bharat/samanantar) (English-Telugu parallel corpus).
2. **Candidate generation:** two Telugu translations per sentence from IndicTrans2 — one via beam search (confident/deterministic), one via temperature sampling (noisier) — to ensure genuine quality contrast between candidates.
3. **Judging:** each pair ranked by an LLM judge (via Groq) using a rubric covering accuracy, naturalness, and grammar.
4. **Manual verification:** 72 pairs (18%) hand-reviewed by a native Telugu speaker. **Disagreement rate: 8.3% (6/72)** — these were corrected before finalizing the dataset. This cross-check is a key methodological strength: most judge-only pipelines have no way to catch a judge rewarding fluent-but-wrong translations.
5. **Final dataset:** 399 preference triples (`{prompt, chosen, rejected}`), split 339 train / 60 held-out.

## Evaluation

- **Win-rate tournament:** every model evaluated against every other model on all 60 held-out prompts (6 matchups × 60 = 360 total judgments), using the same LLM-judge setup as dataset construction. 0 judgments were unparseable.
- **Alignment-tax check:** all four models tested on plain, generic sentences outside the training domain, to confirm none of the preference-optimization methods degraded basic translation ability.

## Discussion — why SFT won

This result is a genuine finding, not a bug — the pipeline ran cleanly (0 unparsed judgments, sensible relative ordering among the three alignment methods themselves: DPO > SimPO > ORPO). The likely explanation:

- **Small dataset (339 training pairs).** Published preference-optimization results are typically demonstrated on tens of thousands of pairs. At this scale, the preference signal may be too sparse for DPO/SimPO/ORPO's losses to reliably outperform straightforward imitation of good examples.
- **Already-specialized base model.** IndicTrans2 is a dedicated MT model, not a general-purpose LLM that badly needs alignment. A model already close to its ceiling on a narrow task leaves less room for preference tuning to help, and more room for it to introduce noise.
- **ORPO's notably high training loss (10.5, vs. DPO's 1.4 and SimPO's 1.0)** — not directly comparable in scale across methods, but combined with ORPO's lowest tournament score, this suggests it may need further hyperparameter tuning to be a fully fair comparison.

## Limitations

- Small base model (200M parameters) and small dataset (339 training pairs) — results may not generalize to larger scales.
- Single LLM judge model for both dataset construction and evaluation (partially mitigated by manual spot-checking).
- Encoder-decoder architecture — `DPOTrainer`/`CPOTrainer`/`ORPOTrainer` are primarily built and tested around decoder-only causal LMs; encoder-decoder support is a less-traveled path.
- No hyperparameter sweep — a single configuration was used per method.

## Repository structure

```
├── notebooks/
│   └── telugu_preference_optimization.ipynb   # full pipeline: data → training → evaluation
├── data/
│   ├── telugu_train.jsonl
│   └── telugu_heldout.jsonl
├── results/
│   └── final_eval_results.json                # win counts, tournament log, alignment-tax check
└── README.md
```

## Stack

`transformers` · `trl` · `peft` · `bitsandbytes` · `datasets` · IndicTransToolkit · Groq API (judge) · Google Colab (T4 GPU)

## What I'd do differently with more time/compute

- Scale the preference dataset to several thousand pairs to test whether the SFT-wins result holds or reverses
- Hyperparameter sweep for ORPO specifically, given its outlier loss
- Add a second, independent judge model to reduce single-judge bias
- Test on a general-purpose multilingual LLM rather than a specialized MT model, to see whether the same result holds when there's more room for alignment to help
