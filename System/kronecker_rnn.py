import torch
from torch import nn


class KfRnnCellFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, h_prev, W_in, W_h, bias, u_prev, A_prev):
        h_q = torch.cat((x, h_prev, torch.ones(1)), dim=-1)

        z_x = x @ W_in
        z_y = h_prev @ W_h
        z = z_x + z_y + bias

        h_next = torch.tanh(z)
        dh_dz = 1 - h_next ** 2
        D = torch.diag(dh_dz)
        H = D @ W_h

        H_q = H @ A_prev

        p1 = (H_q.norm() / (1e-8 + u_prev.norm())) ** 0.5
        p2 = (D.norm() / (1e-8 + h_q.norm())) ** 0.5
        c1 = torch.tensor([-1, 1])[torch.randint(low=0, high=2, size=(1,))]
        c2 = torch.tensor([-1, 1])[torch.randint(low=0, high=2, size=(1,))]

        u_next = (c1 * p1 * u_prev + c2 * p2 * h_q).unsqueeze(0)
        A_next = c1 * (1 / (p1 + 1e-8)) * H_q + (c2 / (p2 + 1e-8)) * D

        ctx.save_for_backward(A_next, u_next, h_next, dh_dz, W_in)

        return h_next

    @staticmethod
    def backward(ctx, grad_output):
        A_next, u_next, h_next, dh_dz, W_in = ctx.saved_tensors

        n_in, n = W_in.shape[0], W_in.shape[1]

        L_grad = torch.kron(u_next, A_next @ grad_output.unsqueeze(1))

        d_a_t = dh_dz * grad_output.unsqueeze(0)
        out_grads_in = d_a_t.mm(W_in.T)
        out_grads_h = None

        L_grad = L_grad.T
        W_in_grad, W_h_grad, b_grad = L_grad[:n_in], L_grad[n_in: n_in + n], L_grad[n + n_in:].squeeze()

        return out_grads_in, out_grads_h, W_in_grad, W_h_grad, b_grad, None, None


class KfRNNCell(nn.Module):
    def __init__(self, input_features, units):
        super(KfRNNCell, self).__init__()
        self.input_features = input_features
        self.units = units

        self.W_in = nn.Parameter(torch.empty(input_features, units, dtype=torch.float32))
        self.W_h = nn.Parameter(torch.empty(units, units, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(units, dtype=torch.float32))

        self.u_prev = torch.zeros(input_features + units + 1, dtype=torch.float32)
        self.A_prev = torch.zeros(units, units, dtype=torch.float32)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.W_in, -0.1, 0.1)
        nn.init.uniform_(self.W_h, -0.1, 0.1)
        nn.init.uniform_(self.bias, -0.1, 0.1)

        self.h_prev = None
        self.buffer = torch.zeros(units)

    def forward(self, inputs):
        if self.buffer is not None:
            self.h_prev = self.buffer

        h_new = KfRnnCellFunction.apply(inputs, self.h_prev,
                                        self.W_in, self.W_h, self.bias,
                                        self.u_prev, self.A_prev)
        self.buffer = h_new.detach()
        return h_new

    def get_hidden(self):
        return self.buffer
