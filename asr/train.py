"""Pasul 5–6 — arhitectura modelului și antrenarea (fine-tuning Wav2Vec2-CTC).

Arhitectura: encoder Wav2Vec2 (CNN pe forma de undă brută -> 24 blocuri Transformer, preantrenat
auto-supravegheat pe 53 de limbi, XLSR-53) + cap liniar CTC peste vocabularul românesc (a-z ă â î ș ț
și spațiu: 32 de simboluri + [UNK] + [PAD]). Encoderul CNN este înghețat; restul se ajustează.

Codul este scris să ruleze și pe transformers 4.x, și pe 5.x (numele argumentelor s-au schimbat).
"""
from __future__ import annotations

import inspect
import json
import time

import numpy as np
import torch

from .config import Config
from .dataset import ASRDataset, DataCollatorCTC, load_split, make_processor
from .evaluate import wer_cer


def build_tiny_model(vocab_size: int, pad_id: int):
    """Model Wav2Vec2-CTC minuscul, inițializat aleator — NUMAI pentru testul rapid ("smoke").
    Nu necesită descărcare și verifică pe CPU că întregul pipeline de antrenare/evaluare rulează."""
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
    conf = Wav2Vec2Config(
        vocab_size=vocab_size, pad_token_id=pad_id, hidden_size=96, num_hidden_layers=3,
        num_attention_heads=4, intermediate_size=192, conv_dim=(48,) * 7,
        conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_feat_extract_layers=7, feat_extract_norm="layer", num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=4, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        mask_time_prob=0.0, layerdrop=0.0, hidden_dropout=0.0, attention_dropout=0.0,
        feat_proj_dropout=0.0, final_dropout=0.0, apply_spec_augment=False,
    )
    return Wav2Vec2ForCTC(conf)


def load_model(cfg: Config, vocab_size: int, pad_id: int, smoke: bool = False):
    from transformers import Wav2Vec2ForCTC
    if smoke:
        return build_tiny_model(vocab_size, pad_id)
    return Wav2Vec2ForCTC.from_pretrained(
        cfg.base_model,
        attention_dropout=0.1, hidden_dropout=0.1, feat_proj_dropout=0.0, final_dropout=0.0,
        mask_time_prob=0.05, layerdrop=0.0 if cfg.gradient_checkpointing else 0.05,
        ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=pad_id, vocab_size=vocab_size, ignore_mismatched_sizes=True)


def _filter_kwargs(cls, kw: dict) -> dict:
    params = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(k for k in kw if k not in params)
    if dropped:
        print(f"  ℹ argumente ignorate de versiunea instalată: {dropped}")
    return {k: v for k, v in kw.items() if k in params}


def make_training_args(cfg: Config, smoke: bool, max_steps: int | None, epochs: float | None):
    from transformers import TrainingArguments
    has_tb = False
    try:
        import tensorboard  # noqa: F401
        has_tb = True
    except ImportError:
        pass
    cuda = torch.cuda.is_available()
    kw = dict(
        output_dir=str(cfg.checkpoints_dir),
        per_device_train_batch_size=cfg.batch_size if not smoke else 8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=cfg.grad_accum if not smoke else 1,
        num_train_epochs=epochs if epochs is not None else cfg.epochs,
        max_steps=max_steps if max_steps else -1,
        learning_rate=cfg.learning_rate if not smoke else 1e-3,
        warmup_ratio=cfg.warmup_ratio if not smoke else 0.05,
        weight_decay=0.0, lr_scheduler_type="linear",
        fp16=cuda and not smoke,
        gradient_checkpointing=cfg.gradient_checkpointing and not smoke,
        logging_steps=10 if smoke else 25,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        report_to=["tensorboard"] if has_tb else "none",
        dataloader_num_workers=0 if smoke else 2,
        remove_unused_columns=False, label_names=["labels"], seed=cfg.seed, use_cpu=not cuda,
    )
    strat = "steps" if max_steps else "epoch"
    name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters \
        else "evaluation_strategy"
    kw[name] = strat
    kw["save_strategy"] = strat
    if max_steps:
        kw["eval_steps"] = kw["save_steps"] = max(1, max_steps // 5)
    return TrainingArguments(**_filter_kwargs(TrainingArguments, kw))


def train(cfg: Config, smoke: bool = False, max_steps: int | None = None, epochs: float | None = None,
          resume: bool = False) -> dict:
    from transformers import EarlyStoppingCallback, Trainer
    cfg.make_dirs()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    train_rows, dev_rows = load_split(cfg, "train"), load_split(cfg, "dev")
    # procesor + model (atenție mask: modelele XLSR/XLS-R folosesc attention_mask, wav2vec2-base nu)
    proc = make_processor(cfg.final_dir, cfg.sampling_rate, True)
    pad_id, vocab_size = proc.tokenizer.pad_token_id, len(proc.tokenizer)
    model = load_model(cfg, vocab_size, pad_id, smoke)
    use_mask = getattr(model.config, "feat_extract_norm", "layer") == "layer"
    if not use_mask:
        proc = make_processor(cfg.final_dir, cfg.sampling_rate, False)
    if cfg.freeze_feature_encoder and not smoke:
        model.freeze_feature_encoder()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 {cfg.base_model if not smoke else 'tiny-random (smoke)'} · parametri antrenabili: {n_train/1e6:.1f}M · "
          f"vocab {vocab_size}")

    train_ds = ASRDataset(train_rows, proc, cfg)
    dev_ds = ASRDataset(dev_rows, proc, cfg)
    print(f"📚 train {len(train_ds)} segmente ({sum(train_ds.lengths)/cfg.sampling_rate/3600:.2f} h) · "
          f"dev {len(dev_ds)}")
    if len(train_ds) == 0 or len(dev_ds) == 0:
        raise RuntimeError("Set de antrenare sau dev gol.")

    collator = DataCollatorCTC(pad_id, use_mask)

    def preprocess_logits(logits, labels):
        return torch.argmax(logits, dim=-1)          # evită stocarea logiturilor complete

    def compute_metrics(pred):
        ids = np.asarray(pred.predictions)
        lab = np.asarray(pred.label_ids).copy()
        lab[lab == -100] = pad_id
        hyps = proc.batch_decode(ids)
        refs = proc.tokenizer.batch_decode(lab, group_tokens=False)
        m = wer_cer(refs, hyps)
        return {"wer": m["wer"] if m["wer"] is not None else 1.0, "cer": m["cer"] or 0.0}

    class CTCTrainer(Trainer):
        """Grupează exemplele de lungime similară (mai puțin padding)."""
        def _get_train_sampler(self, *a, **k):
            try:
                from transformers.trainer_pt_utils import LengthGroupedSampler
                return LengthGroupedSampler(self.args.train_batch_size, dataset=self.train_dataset,
                                            lengths=self.train_dataset.lengths)
            except Exception:
                return super()._get_train_sampler(*a, **k)

    args = make_training_args(cfg, smoke, max_steps, epochs)
    cb = [] if smoke else [EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)]
    tkw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds, data_collator=collator,
               compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_logits, callbacks=cb)
    trainer = CTCTrainer(**_filter_kwargs(Trainer, tkw))

    t0 = time.time()
    out = trainer.train(resume_from_checkpoint=True if resume else None)
    dev = trainer.evaluate()
    trainer.save_model(str(cfg.final_dir))
    proc.save_pretrained(str(cfg.final_dir))

    summary = {
        "base_model": "tiny-random (smoke)" if smoke else cfg.base_model,
        "train_segments": len(train_ds), "dev_segments": len(dev_ds),
        "train_hours": round(sum(train_ds.lengths) / cfg.sampling_rate / 3600, 3),
        "train_loss": round(float(out.training_loss), 4), "global_step": int(out.global_step),
        "dev_wer": dev.get("eval_wer"), "dev_cer": dev.get("eval_cer"), "dev_loss": dev.get("eval_loss"),
        "train_minutes": round((time.time() - t0) / 60, 2), "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hyperparameters": {k: getattr(cfg, k) for k in (
            "epochs", "batch_size", "grad_accum", "learning_rate", "warmup_ratio",
            "early_stopping_patience", "gradient_checkpointing", "freeze_feature_encoder", "seed")},
        "log_history": trainer.state.log_history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (cfg.final_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Antrenare gata în {summary['train_minutes']} min · dev WER {summary['dev_wer']} · "
          f"model → {cfg.final_dir}")
    return summary
