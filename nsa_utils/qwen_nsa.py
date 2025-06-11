import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm
from deepspeed.pipe import PipelineModule

from typing import Tuple
from native_sparse_attention.modeling_nsa import NativeSparseAttention 

class Qwen2DecoderLayerPipe(nn.Module):
    def __init__(self, decoder_layer: Qwen2DecoderLayer):
        super().__init__()
        self.decoder_layer =  decoder_layer

    def forward(self, inputs: Tuple):
        hidden_states = inputs[0]
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        outputs = self.decoder_layer(
            hidden_states = hidden_states,
            use_cache = True
        )
        return outputs

class EmbeddingLayerPipe(nn.Module):
    def __init__(self, embed_layer: nn.Embedding):
        super().__init__()
        self.embed_layer = embed_layer

    def forward(self, inputs: Tuple):
        input_ids, _ = inputs
        return (self.embed_layer(input_ids),)

class LossFuncLayerPipe(nn.Module):
    def __init__(self, norm: Qwen2RMSNorm, lm_head: nn.Linear):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
        
    def forward(self, inputs: Tuple):
        hidden_states = inputs[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.transpose(0, 1).unsqueeze(0)
        logits.requires_grad_(True)

        return logits.transpose(0, 1).unsqueeze(0)
    
class NsaQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        for i, layer in enumerate(self.model.layers):
            attn = NativeSparseAttention(
                hidden_size = 896,
                num_heads = 64,
                num_kv_heads = 4,
                head_dim = 64,
                block_size = 64,
                block_counts = 16,
                window_size = 512,
                layer_idx = i,
            )
            layer.self_attn = attn


    def to_layers(self):
        layers = [
            EmbeddingLayerPipe(self.model.embed_tokens),
            *[Qwen2DecoderLayerPipe(layer) for layer in self.model.layers],
            LossFuncLayerPipe(
                self.model.norm,
                self.lm_head
            )
        ]
        return layers


def build_pipeline_qwen2(model: nn.Module, num_stages: int = 1):
    return PipelineModule(
        layers=model.to_layers(),
        num_stages=num_stages,
        activation_checkpoint_interval=1,
        loss_fn=nn.CrossEntropyLoss(ignore_index=-100),
        checkpointable_layers=model.to_layers()[1:]
    )

def load_custom_weights_and_freeze(model: nn.Module, pretrained_state_dict: dict):
    for name, param in model.named_parameters():
        if name in pretrained_state_dict and param.shape == pretrained_state_dict[name].shape:
            param.data.copy_(pretrained_state_dict[name])
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"{name} not loaded, set to trainable.")
    return model
