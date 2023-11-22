import math
from typing import Optional

import torch
import diffusers
from diffusers.models import attention_processor
import torchvision.transforms.functional as TF


class SelfGuidanceAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: attention_processor.Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform attention-related computations for self-guidance.

        Args:
            attn: diffusers.models.attention_processor.Attention) - Attention instance.
            hidden_states: torch.Tensor - Input tensor.
            encoder_hidden_states: Optional[torch.Tensor] Encoder hidden states tensor.
            attention_mask: Optional[torch.Tensor] - Attention mask tensor.
            temb: Optional[torch.Tensor] - Optional temperature tensor for attention normalization.

        Returns:
            torch.Tensor: Processed hidden states after attention computations.
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        ### SELF GUIDANCE
        scores_ = attention_probs.reshape(len(query), -1, *attention_probs.shape[1:]).mean(1)
        h=w=math.isqrt(scores_.shape[1])
        scores_ = scores_.reshape(len(scores_), h, w, -1)
        global _SG_RES
        if _SG_RES != scores_.shape[2]:
          scores_ = TF.resize(scores_.permute(0,3,1,2), _SG_RES, antialias=True).permute(0,2,3,1)
        try:
          global save_aux
          if not save_aux:
            len_=len(attn._aux['attn'])
            del attn._aux['attn']
            attn._aux['attn'] = [None]*len_ + [scores_]
          else:
            attn._aux['attn'][-1] = attn._aux['attn'][-1].cpu()
            attn._aux['attn'].append(scores_)
        except:
          del attn._aux['attn']
          attn._aux = {'attn': [scores_]}
        ### END SELF GUIDANCE

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
