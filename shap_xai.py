import shap
import torch
import matplotlib.pyplot as plt


def explain_shap(model, data_loader):
    model.eval()

    # print(len(list(data_loader)))
    # Assuming your data_loader returns (anchors, positives, negatives)
    for anchors, positives, negatives in data_loader:
        anchors = anchors.unsqueeze(0)  # Add batch dimension
        positives = positives.unsqueeze(0)
        negatives = negatives.unsqueeze(0)

        # Create an explainer
        explainer = shap.Explainer(model, anchors)

        # Compute SHAP values
        shap_values = explainer.shap_values(positives)

        # Visualize SHAP values
        shap.plots.image_plot(shap_values, positives)
        plt.show()

# Note: You might need to adapt the data loading and processing according to your model's requirements.


def explain_shap_2(model, data_loader):
    model.eval()

    # Create an explainer
    explainer = shap.Explainer(model.forward_once, data_loader)

    # Compute SHAP values for the test data
    shap_values = explainer(data_loader)

    # Visualize SHAP values (you can adjust this part according to your needs)
    shap.plots.image_plot(shap_values, data_loader)
