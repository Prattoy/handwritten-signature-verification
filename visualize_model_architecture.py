import torch
from torchviz import make_dot


def vis_model_arc(model, batch_size, image_size):

    # Dummy input
    dummy_input = torch.randn(batch_size, 1, image_size)

    # Generate the computation graph
    output = model.forward(dummy_input, dummy_input, dummy_input)  # Perform a forward pass
    graph = make_dot(output, params=dict(model.named_parameters()))

    # Save the graph to a file
    graph.render("siamese_network", format="png")