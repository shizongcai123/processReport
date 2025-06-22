import torch
from torchview import draw_graph
from B3_imp import DeeperCNN

model = DeeperCNN(in_channels=1, num_classes=10)
model.eval()

graph = draw_graph(
    model,
    input_size=(1, 1, 28, 28),
    expand_nested=True,
    save_graph=True,
)
graph.visual_graph.attr(size="10,5")  
graph.visual_graph.render("./deeper_cnn_view", format="svg")
