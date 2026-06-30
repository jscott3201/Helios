# 06 — KV Cache, SSD Offload, and Compression

## Design rule

Do not start with live dense-weight SSD paging. Use SSD to persist and restore inactive prefix/KV blocks so repeated long prompts avoid recomputation.

## Cache stages

### Stage A — Active BF16 KV

- Native MLX side owns active KV tensors.
- Rust owns logical cache IDs and metrics.
- No RAM prefix cache, SSD, or compression.

### Stage B — RAM prefix cache

- Block-based cache, e.g. 1024-token blocks.
- Prefix sharing and copy-on-write.
- Exact restore validation against fresh prefill.
- LRU eviction based on memory budget.

### Stage C — SSD cold prefix cache

- Persist inactive prefix blocks to SSD.
- Use safetensors-compatible format first or a simple packed internal format with checksums.
- Restore before prefill only in MVP.
- No mid-decode SSD fetch in MVP.

### Stage D — Compression

Initial compression sequence:

```text
1. BF16 active KV baseline.
2. MLX affine q8 RAM prefix cache.
3. MLX affine q4 SSD prefix cache.
4. Planar4/Planar3 K-only global prefix cache experiments.
5. Iso4/Iso3 symmetric experiments.
6. Fused Metal decode over compressed global K only if evidence supports it.
```

## Gemma 4-specific layout

Gemma 4 has sliding/local attention and periodic full/global attention. Cache metadata must distinguish:

```text
absolute sequence position
block-local position
sliding-window-local position
full-attention cumulative length
layer attention type
logical K/V sharing
physical stored tensor count
head_dim and kv_heads
compression type
```

## Cache key fields

Every cache namespace must include:

```text
model repo/revision/weight hash
quantization hash
tokenizer hash
chat template hash
prompt token-prefix hash
raw prompt hash
adapter id and adapter hash for standard LoRA
KV layout version
KV dtype/compression mode
MLX version
engine version
```

## Acceptance

A cache mode is accepted only when:

- restored logits match fresh-prefill logits for that same mode,
- wrong cache namespaces are rejected,
- corrupted blocks are rejected,
- memory accounting is reported,
- benchmark report shows cold vs warm TTFT.
