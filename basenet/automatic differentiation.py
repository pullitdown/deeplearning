#conding=gbk
import torch

a=torch.arange(4.0)
a.requires_grad_(True)
print(a.grad)
b=torch.arange(4,8,dtype=torch.float32)
b.requires_grad_(True)
y1=torch.dot(b,b)
y1.backward()
print(b.grad)
y=2*torch.dot(a,b)
print(a.grad)

print(y)
y.backward()
print(a.grad)
print(b.grad)

a.grad.zero_()
y2=a.sum()
y2.backward()
print(a.grad)
print(a.data)
a.grad.zero_()
y2=torch.dot(a,a)
y2.backward()#only trough backward the grad in x would be add up
print(a.grad)