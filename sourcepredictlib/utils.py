def print_class(classes, pred):
    [print(f'{i}:{j}') for i, j in zip(list(classes), list(pred[0, :]))]
