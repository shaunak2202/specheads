"""EAGLE-style drafter: predict the target's next hidden feature, then decode it.

One Qwen2 decoder layer takes ``concat(feature_t, embed(token_{t+1}))``, projects
it back to model width, and predicts ``feature_{t+1}``. Tokens come from pushing
that predicted feature through the frozen LM head.

The contrast with Medusa is the point of having both. Medusa's K heads read one
hidden state in parallel, so drafting K tokens costs one small forward but each
head predicts its horizon independently and without seeing what the earlier heads
chose. EAGLE drafts **autoregressively** -- depth d+1 is conditioned on the
feature and token predicted at depth d -- which is strictly more informed and
strictly more expensive, since the drafter's latency lands on the critical path
once per depth rather than once per step.

The LM head and token embeddings are the frozen target's, and on Qwen2.5 they are
the same tied matrix. Neither is trained here.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from ..decode.speculative import DraftContext
from ..decode.tree import TreeSpec


class EagleDrafter(nn.Module):
    """Single-layer autoregressive feature predictor."""

    def __init__(self, config, intermediate_size: int | None = None) -> None:
        super().__init__()
        import copy

        layer_config = copy.deepcopy(config)
        if intermediate_size is not None:
            # The MLP is ~41M of the layer's ~47M params, so this is the only
            # real size lever if drafter latency starts eating the speedup.
            layer_config.intermediate_size = intermediate_size
        layer_config._attn_implementation = "sdpa"

        self.hidden_size = layer_config.hidden_size
        # concat(feature, embedding) -> hidden
        self.fusion = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.layer = Qwen2DecoderLayer(layer_config, layer_idx=0)
        self.config = layer_config

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        features: torch.Tensor,
        token_embeddings: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the next feature for each position.

        Args:
            features: ``[batch, seq, hidden]`` target features at t.
            token_embeddings: ``[batch, seq, hidden]`` embeddings of token t+1.
            position_embeddings: RoPE ``(cos, sin)``, taken from the target's own
                rotary module so the drafter shares the target's position basis.
            attention_mask: optional 4D mask; SDPA's causal flag is disabled
                whenever a mask is supplied.

        Returns:
            ``[batch, seq, hidden]`` predicted features at t+1.
        """
        fused = self.fusion(torch.cat([features, token_embeddings], dim=-1))
        return self.layer(
            fused,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=False,
        )


class EagleTreeDrafter:
    """Expands a `TreeSpec` by running the EAGLE drafter depth by depth.

    Siblings at a depth are distinct because they are different ranks of the same
    top-k, which is the invariant `accept_path` requires. Nodes at the same depth
    are batched into one drafter forward, so the cost is one small forward per
    *depth* rather than per node.
    """

    def __init__(self, drafter: EagleDrafter, target) -> None:
        self.drafter = drafter
        self.target = target
        self.embed = target.model.model.embed_tokens
        self.lm_head = target.lm_head
        self.rotary = target.model.model.rotary_emb

    @torch.no_grad()
    def draft(self, spec: TreeSpec, context: DraftContext) -> torch.Tensor:
        device = context.hidden.device
        drafter_dtype = next(self.drafter.parameters()).dtype
        head_dtype = next(self.lm_head.parameters()).dtype

        tokens = torch.zeros(spec.size, dtype=torch.long)
        # Per node: the feature predicted *at* that node, used to expand its children.
        node_features: dict[int, torch.Tensor] = {}

        by_depth: dict[int, list[int]] = {}
        for node, path in enumerate(spec.ordered):
            by_depth.setdefault(len(path), []).append(node)

        for depth in sorted(by_depth):
            nodes = by_depth[depth]
            parent_features = []
            parent_tokens = []
            for node in nodes:
                parent = spec.parents[node]
                if parent < 0:
                    parent_features.append(context.hidden)
                    parent_tokens.append(context.pending_token)
                else:
                    parent_features.append(node_features[parent])
                    parent_tokens.append(int(tokens[parent]))

            features = torch.stack(parent_features).to(drafter_dtype).unsqueeze(1)
            token_ids = torch.tensor(parent_tokens, dtype=torch.long, device=device).unsqueeze(1)
            embeddings = self.embed(token_ids).to(drafter_dtype)

            positions = torch.full(
                (features.shape[0], 1), depth - 1, dtype=torch.long, device=device
            )
            rope = self.rotary(features, positions)
            predicted = self.drafter(features, embeddings, rope)[:, 0]

            logits = self.lm_head(predicted.to(head_dtype)).float()
            max_rank = max(spec.ordered[n][-1] for n in nodes)
            ranked = torch.topk(logits, k=max_rank + 1, dim=-1).indices

            for row, node in enumerate(nodes):
                tokens[node] = ranked[row, spec.ordered[node][-1]]
                node_features[node] = predicted[row]

        return tokens
