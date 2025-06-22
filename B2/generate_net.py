import torch
from torchview import draw_graph
from similar_pic import SiameseNetwork

model = SiameseNetwork()

model.eval()

graph = draw_graph(
    model,
    input_size=[(1, 1, 28, 28), (1, 1, 28, 28)],
    expand_nested=True,
)
graph.visual_graph.render("./siamese_network_view", format="svg")