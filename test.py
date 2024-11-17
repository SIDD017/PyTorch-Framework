import numpy as np
import custom_torch as torch
import time

x1 = torch.Tensor([1,2,3], "cuda")
x2 = torch.Tensor([[1,2,3],[4,7,6], [7,3,2]], "cuda")
x3 = torch.Tensor([[[1,2,3],[4,7,6]], [[7,3,2], [3,8,0]], [[5,6,7], [4,8,0]]], "cuda")

print("x1")
x1.print()
print("x2")
x2.print()
print("x3")
x3.print()

print("Transpose")
x2.transpose().print()

print("neg")
x2.neg().print()

print("reciprocal")
x2.reciprocal()

print("add")
x2.add(x2).print()

print("subtract")
x2.subtract(x2).print()

print("mult")
x2.mult(10).print()

print("elementwise_mult")
x2.elementwise_mult(x2).print()

print("pow")
x2.pow(3).print()

print("relu")
x2.relu().print()

print("binarilize")
x2.binarilize().print()

print("exp")
x2.exp().print()


print("MOVING TO CUDA")
x1.to("cuda")
x2.to("cuda")
x3.to("cuda")

print("x1")
x1.print()
print("x2")
x2.print()
print("x3")
x3.print()

print("Transpose")
x2.transpose().print()

print("neg")
x2.neg().print()

print("reciprocal")
x2.reciprocal()

print("add")
x2.add(x2).print()

print("subtract")
x2.subtract(x2).print()

print("mult")
x2.mult(10).print()

print("elementwise_mult")
x2.elementwise_mult(x2).print()

print("pow")
x2.pow(3).print()

print("relu")
x2.relu().print()

print("binarilize")
x2.binarilize().print()

print("exp")
x2.exp().print()


