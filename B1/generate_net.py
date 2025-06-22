import torch
from torchview import draw_graph
from resnet_recon import DenoisingResNetSkip, ResBlock

# 1. First draw the high-level architecture without block details
model = DenoisingResNetSkip()
graph_high_level = draw_graph(
    model, 
    input_size=(1, 1, 28, 28),
    expand_nested=False,  # This will hide the details of ResBlock
    depth=1,             # This controls how deep to go in the hierarchy
    hide_inner_tensors=True,
    hide_module_functions=True,
)
graph_high_level.visual_graph.render("./denoising_resnet_skip_high_level", format="svg")

# 2. Then draw just the ResBlock details separately
resblock = ResBlock(1, 32)  # Example with some channels
graph_block = draw_graph(
    resblock,
    input_size=(1, 1, 28, 28),
    expand_nested=True,
    depth=3,
    hide_inner_tensors=False,
)
graph_block.visual_graph.render("./resblock_detail", format="svg")