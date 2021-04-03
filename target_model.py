import torch

class Shapeshifter(torch.nn.Module):
    def __init__(self, in_shape, out_shape):
        super(Shapeshifter, self).__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

    def forward(self, x):
        return x.view((x.size(0), *self.out_shape))

class MLP(torch.nn.Module):
    """
    A simple MLP
    """
    def __init__(self, input_shape, num_classes):
        super(MLP, self).__init__()

        assert len(input_shape) == 3

        c, h, w = input_shape

        self.blocks = (
            Shapeshifter((c, h, w), (c*h * w,)),
            torch.nn.Linear(c * h * w, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

        self.model = torch.nn.Sequential(*self.blocks)

        self.probablize = torch.nn.Softmax(-1)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        x_tensor = torch.tensor(x, dtype= torch.float)
        probs= self.probablize(self.model(x_tensor))
        return probs

if __name__ == '__main__':
    model = MLP((1, 28, 28), 10)
    image = torch.ones((1,1,28,28))
    pred = model(image)
    print(pred.shape)

