def dice_coefficient(labels, predictions):
    return 2 * (labels * predictions).sum() / (labels.sum() + predictions.sum() + 1e-6)
