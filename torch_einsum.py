import torch
import numpy as np

# https://pytorch.org/docs/stable/generated/torch.einsum.html



# diagonal
print(torch.einsum('ii->i', torch.randn(4, 4)))

# outer product
x = torch.randn(5)
y = torch.randn(4)
print(torch.einsum('i,j->ij', x, y))


# batch matrix multiplication
As = torch.randn(3,2,5)
Bs = torch.randn(3,5,4)
print(torch.einsum('bij,bjk->bik', As, Bs))


# batch permute
A = torch.randn(2, 3, 4, 5)
print(torch.einsum('...ij->...ji', A).shape)


# # equivalent to torch.nn.functional.bilinear
# >>> A = torch.randn(3,5,4)
# >>> l = torch.randn(2,5)
# >>> r = torch.randn(2,4)
# >>> torch.einsum('bn,anm,bm->ba', l, A, r)
# tensor([[-0.3430, -5.2405,  0.4494],
#         [ 0.3311,  5.5201, -3.0356]])