import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'native_sparse_attention_pytorch')))
from native_sparse_attention_pytorch.native_sparse_attention import SparseAttention

if __name__ == "__main__":
	attn = SparseAttention(
		dim = 512,
		dim_head = 64,
		heads = 8,
		sliding_window_size = 2,
		compress_block_size = 4,
		compress_block_sliding_stride = 2,
		selection_block_size = 4,
		num_selected_blocks = 2
	)

	tokens = torch.randn(2, 31, 512)
 
	attended = attn(tokens)
	# assert tokens.shape == attended.shape
	print(tokens[0].shape)