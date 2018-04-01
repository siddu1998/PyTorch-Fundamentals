#  gradient function chala pedaga untay mari danini manam emi cheylemu
# so we use predefined framework tools to compute gradient
# in pytroch, pytorch ki tagina variable type lo data teskoravali
import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


#this will draw the forward graph

#this in brief is telling compute the weights using gradient automatically by drwaing a graph
w = Variable(torch.Tensor([1.0]),  requires_grad=True)  #the params ikkada tell it required_grad


def forward(x):
    return x * w




def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

print("Before update",  4, forward(4))

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        #loss computed
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)",  4, forward(4).data[0])
