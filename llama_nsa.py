import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
from accelerate import dispatch_model, infer_auto_device_map
from deepspeed.pipe import PipelineModule

from typing import Tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'native_sparse_attention_pytorch')))
from native_sparse_attention_pytorch.native_sparse_attention import SparseAttention

class LlamaDecoderLayerPipe(nn.Module):
    def __init__(self, decoder_layer: LlamaDecoderLayer):
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
    def __init__(self, norm: LlamaRMSNorm, lm_head: nn.Linear):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
        
    def forward(self, inputs: Tuple):
        hidden_states = inputs[0]
        hidden_states = self.norm(hidden_states)
        slice_indices = slice(-1, None)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits.squeeze(0)
    
class NsaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        for i, layer in enumerate(self.model.layers):
            attn = SparseAttention(
                dim = 4096,
                dim_head = 128,
                heads = 32,
                sliding_window_size = 512,
                compress_block_size = 32,
                compress_block_sliding_stride = 32,
                selection_block_size = 32,
                num_selected_blocks = 16,
                kv_heads = 8,
                layer_idx = i,
                use_diff_topk = True,
            )
            layer.self_attn = attn

    def to_layers(self):
        layers = [
            EmbeddingLayerPipe(self.model.embed_tokens),
            *[LlamaDecoderLayerPipe(layer) for layer in self.model.layers],
            LossFuncLayerPipe(
                self.model.norm,
                self.lm_head
            )
        ]
        return layers


def build_pipeline_llama(model: nn.Module, num_stages: int = 1):
    return PipelineModule(
        layers=model.to_layers(),
        num_stages=num_stages,
        activation_checkpoint_interval=1,
        loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
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


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    config = LlamaConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pretrained_llama_model = LlamaForCausalLM.from_pretrained(model_id)
    pretrained_state_dict = pretrained_llama_model.state_dict()
    model = NsaLlamaForCausalLM(config)
    model = load_custom_weights_and_freeze(model, pretrained_state_dict)
    device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"])
    model = dispatch_model(model, device_map=device_map)

