#coding=gbk

from numpy.core.fromnumeric import shape
from numpy.core.numeric import NaN
from numpy.lib.function_base import copy
import sklearn as skl
import torch
ones=torch.ones((5,4))
print(ones)
x=torch.arange(14).reshape((2,7))
print(x)
import pandas as pd
input=pd.DataFrame({'a':[1,2,3],'b':['llloc',NaN,'pppp']})
input=pd.get_dummies(input,dummy_na=True)
print(input)

k=torch.arange(12).reshape((3,4))
p=k.clone()
print(type(k),type(p))
print(torch.cat((k,p),dim=0))
print(torch.cat((k,p),dim=1))

ten=torch.arange(40,dtype=torch.float32).reshape((2,4,5))
print(ten.mean(axis=0))
print(ten.sum(axis=0)/ten.shape[0])
print(ten.mean(axis=1))
print(ten/ten.sum(axis=1,keepdims=True))#后缀这个keepdims十分重要,因为决定的矩阵的形状不至于改变


print(ten.sum(axis=[0,2],keepdims=True))
#dot
pp=torch.ones(4,dtype=torch.float32)
kk=torch.arange(4,dtype=torch.float32)
print(pp.size(),kk.size())
print(kk*pp)
print(torch.dot(kk,pp))
print(torch.sum(pp*kk))

mat1=torch.arange(12,dtype=torch.float32).reshape((3,4))
print(torch.mv(mat1,pp))

mat2=torch.arange(20,dtype=torch.float32).reshape((4,5))
print(torch.mm(mat1,mat2))

#范数
#L2范数||X||2=sqrt(sum(x^2))
u=torch.Tensor([3.0,-4.0])
L2=torch.norm(u)
print(L2)
#L1范数||X||1=torch.abs(u).sum()
L1=torch.abs(u).sum()