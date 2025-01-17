from torch import nn, topk, nonzero, zeros, where, float

class Model(nn.Module):
    def __init__(self, lenet, resnet):
        super(Model, self).__init__()
        self.lenet_model = lenet
        self.resnet_model = resnet
        self.conf_matrix = zeros(10, 10)
        self.lenet_calls = 0
        self.resnet_calls = 0

    def eval(self):
        self.lenet_model.eval()
        self.resnet_model.eval()

    def forward(self, x, threshold, labels):
        out = self.lenet_model(x)
        self.lenet_calls += len(out)

        pred = out.argmax(dim=1)
        for i in range(10):
            for j in range(10):
                matches = where((pred == j) & (labels == i))[0]
                self.conf_matrix[i, j] += len(matches)

        probs = nn.Softmax(dim=1)(out)
        vals, _ = topk(probs, 2, dim=1)
        diff = vals[:, 0] - vals[:, 1]
        inds = nonzero(diff <= threshold).squeeze()

        if inds.numel() == 0:
            return out
        if inds.ndim == 0:
            self.resnet_calls += 1
            out[inds.item()] = self.resnet_model(x[inds.item()].unsqueeze(0))
        else:
            for i in inds:
                self.resnet_calls += 1
                out[i.item()] = self.resnet_model(x[i.item()].unsqueeze(0))

        return out