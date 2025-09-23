from typing import Optional
import torch
import torch.nn.functional as F 
from torch import nn 
import math

# source: https://nn.labml.ai/diffusion/stable_diffusion/model/unet_attention.html
# flash attention: https://github.com/Dao-AILab/flash-attention

class SpatialTransformer(nn.Module):
    def __init__(self,channels:int,n_heads:int,d_head:int,depth:int=1,context_dim=None):
        super().__init__()
        self.norm = normalization(channels) # Normalization layer
        self.proj_in = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0) # 1x1 convolution
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels,n_heads,d_head,context_dim=context_dim) for _ in range(depth)])
        self.proj_out = nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)

    def _preprocess(self,x:torch.Tensor):
        b,c,h,w = x.shape
        x_in = x 
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0,2,3,1).view(b,h*w,c) # Transpose and reshape from [batch_size, channels, height, width] to [batch_size, height * width, channels]
        return x,(x_in,b,h,w,c) # Return the reshaped tensor and the original shape parameters
    
    def forward(self,x:torch.Tensor,cond=None):
        x,(x_in,b,h,w,c) = self._preprocess(x) # Preprocess input tensor
        for block in self.transformer_blocks:
            x = block(x,cond)
        x = x.view(b,h,w,c).permute(0,3,1,2) # Reshape and transpose from [batch_size, height * width, channels] to [batch_size, channels, height, width]
        x = self.proj_out(x)
        return x + x_in


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class BasicTransformerBlock(nn.Module):
    def __init__(self,d_model:int,n_heads:int,d_head:int,context_dim=None):
        super().__init__()
        self.attn1 = CrossAttention(d_model=d_model,d_cond=d_model,n_heads=n_heads,d_head=d_head)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = CrossAttention(d_model=d_model,d_cond=context_dim,n_heads=n_heads,d_head=d_head)
        self.norm2 = nn.LayerNorm(d_model) # cross-attention normalization

        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model) # feed-forward normalization

    def forward(self, x, context=None):
        return self._forward(x, context=context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond = context) + x
        x = self.ff(self.norm3(x)) + x
        return x



class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)



class CrossAttention(nn.Module):
    use_flash_attention: bool = False
    def __init__(self,d_model:int,d_cond:int,n_heads:int,d_head:int,is_inplace:bool=True):
        super().__init__()
        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head** -0.5 # attention scaling factor
        d_attn = d_head * n_heads 
        self.to_q = nn.Linear(d_model,d_attn,bias=True)
        self.to_k = nn.Linear(d_cond,d_attn,bias=False)
        self.to_v = nn.Linear(d_cond,d_attn,bias=False) # query, key and value mappings

        self.to_out = nn.Sequential(nn.Linear(d_attn,d_model))

        try:
            from flash_attn.flash_attention import FlashAttention # type: ignore
            self.flash = FlashAttention()
            self.flash.softmax_scale = self.scale
        except ImportError:
            self.flash = None
        
    def forward(self,x:torch.Tensor,cond:Optional[torch.Tensor]=None):
        has_cond = cond is not None
        if not has_cond:
            cond = x # if cond is None we perform self-attention
        
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        if CrossAttention.use_flash_attention and self.flash is not None and not has_cond and self.d_head <= 128:
            return self.flash_attention(q,k,v) # use flash attention if it's available and the head size is less or equal to 128
        else:
            return self.normal_attention(q,k,v)

    def flash_attention(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        batch_size, seq_len,_ = q.shape
        qkv = torch.stack((q,k,v),dim=2)# Stack q , k , v vectors for flash attention, to get a single tensor of shape [batch_size, seq_len, 3, n_heads * d_head]
        qkv = qkv.view(batch_size,seq_len,3,self.n_heads,self.d_head) # split the heads
        if self.d_head <=32: # Flash attention works for head sizes 32 , 64 and 128 , so we have to pad the heads to fit this size.
            pad = 32 - self.d_head
        elif self.d_head <=64:
            pad = 64 - self.d_head
        elif self.d_head <=128:
            pad = 128 - self.d_head
        else:
            raise ValueError(f'Head size ${self.d_head} too large for Flash Attention') 
        if pad :
            qkv = torch.cat((qkv,qkv.new_zeros(batch_size,seq_len,3,self.n_heads,3)),dim=-1) # pad the heads
        out,_ = self.flash(qkv) # compute attention; This gives a tensor of shape [batch_size, seq_len, n_heads, d_padded]
        out = out[:,:,:,:self.d_head] # truncate the extra head size
        out = out.reshape(batch_size,seq_len,self.n_heads*self.d_head) # Reshape to [batch_size, seq_len, n_heads * d_head]
        return self.to_out(out)

    def normal_attention(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        # q,k,v are the query vectors before splitting heads, of shape [batch_size, seq, d_attn]
        q = q.view(*q.shape[:2],self.n_heads,-1)
        k = k.view(*k.shape[:2],self.n_heads,-1)
        v = v.view(*v.shape[:2],self.n_heads,-1) # Split them to heads of shape [batch_size, seq_len, n_heads, d_head]

        attn = torch.einsum('bihd,bjhd->bhij',q,k) * self.scale # calculate attention

        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] - attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1) # compute softmax

        out = torch.einsum('bhij,bjhd->bihd',attn,v) # compute attention output

        out = out.reshape(*out.shape[:2],-1) # Reshape to [batch_size, height * width, n_heads * d_head]
        return self.to_out(out)

class FeedForward(nn.Module):

    def __init__(self,d_model:int,d_mult:int=4):
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model,d_model*d_mult),
            nn.Dropout(0.),
            nn.Linear(d_model*d_mult,d_model)
        )
    def forward(self,x:torch.Tensor):
        return self.net(x)

class GeGLU(nn.Module):
    '''
    GeGLU Activation
    GeGLU(x)=(xW+b)∗GELU(xV+c)
    '''
    def __init__(self,d_in:int,d_out:int):
        super().__init__()

        self.proj = nn.Linear(d_in,d_out*2) # Combined linear projections xW+b and xV+c

    def forward(self,x:torch.Tensor):
        x, gate = self.proj(x).chunk(2,dim=-1)
        return x * F.gelu(gate)



def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)
    

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads