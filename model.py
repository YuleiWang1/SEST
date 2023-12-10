import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
import numpy as np
from einops import rearrange, repeat
from CA import *


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


#########################################
########### window operation#############
def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.weight_factor = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.weight_factor = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=False)
        # self.ca_atten = SELayer1d(48)
        self.ca_atten = SE(48)
        # self.ca_atten = CAMA(48)
        # self.ca_atten = CABlock1d(48)
        # self.ca_atten = CAB(48)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape

        shortcut = x
        # print('145_{}'.format(shortcut.shape))
        # print('145_{}'.format(x.shape))

        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # print('183_{}'.format(x.shape))
        # print(self.weight_factor)

        x = x + self.weight_factor * self.ca_atten(shortcut)

        # print('187_{}'.format(x.shape))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):

        # print('198:{}'.format(x.shape))

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        # print('204:{}'.format(x.shape))

        x = self.fc2(x)
        x = self.drop2(x)

        # print('209:{}'.format(x.shape))

        return x


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):

        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        # hidden_features = hidden_features

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)

        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2

        x = self.project_out(x)

        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_mlp='mlp'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if token_mlp in ['ffn', 'mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'gdfn':
            self.mlp = FeedForward(dim, ffn_expansion_factor=2.66)
        else:
            raise Exception("FFN error!")

        # self.weight_factor = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        # self.weight_factor = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=False)
        # self.ca_atten = SELayer(48)
        # self.ca_atten = CAB(48)
        # self.ca_atten = CABlock(48)
        # self.ca_atten = SE2d(48)


    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # print(x.shape, attn_mask.shape)

        # print('window', self.shift_size, self.window_size)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cab = x

        # print('410_{}'.format(x.shape))

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # print('424_{}'.format(x.shape))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # print(attn_mask)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # print("422_{}".format(shifted_x.shape))

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        # print("434_{}".format(x.shape))
        # print(self.weight_factor)

        # x = x + self.weight_factor * self.ca_atten(cab)

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, token_mlp='mlp', shift_flag=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, token_mlp=token_mlp)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, token_mlp=token_mlp)
                for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):

        # print(self.shift_size, self.window_size,  x.shape)

        res = x
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]

        # print(attn_mask, attn_mask.shape)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        out = res + x

        return out, H, W
        # return out


class SEST(nn.Module):
    def __init__(self):
        super(SEST, self).__init__()
        num_channel = 31
        num_feature = 48
        depth = 6
        token_mlp = 'leff'
        shift_flag = False
        # depth = 8

        # self.SF = nn.Conv2d(num_channel, num_feature, 3, 1, 1)
        self.SF = nn.Conv2d(num_channel+3, num_feature, 3, 1, 1)

        # self.Embedding = nn.Sequential(
        #     nn.Linear(num_channel+3, num_feature),
        # )

        self.trans2 = BasicLayer(dim=num_feature, depth=depth, num_heads=3, window_size=8, mlp_ratio=4, token_mlp=token_mlp, shift_flag=shift_flag)
        self.trans4 = BasicLayer(dim=num_feature, depth=depth, num_heads=3, window_size=16, mlp_ratio=4, token_mlp=token_mlp, shift_flag=shift_flag)
        self.trans8 = BasicLayer(dim=num_feature, depth=depth, num_heads=3, window_size=32, mlp_ratio=4, token_mlp=token_mlp, shift_flag=shift_flag)
        # 单窗口
        # self.trans = BasicLayer(dim=num_feature, depth=depth, num_heads=3, window_size=20, mlp_ratio=4, token_mlp=token_mlp, shift_flag=shift_flag)

        # self.re = nn.Sequential(
        #     nn.Conv2d(3*num_feature, num_feature, 3, 1, 1),
        #     nn.LeakyReLU()
        # )

        self.refine = nn.Sequential(
            nn.Conv2d(num_feature, num_feature, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_feature, num_channel, 3, 1, 1)
        )

        # self.ca_attn = ChannelAttention(31)

        # self.weight_factor = torch.nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)

    def forward(self, HSI, MSI):
        # ca = self.ca_attn(HSI)
        # print(np.shape(ca))
        # R = 26
        # G = 14
        # B = 6

        UP_LRHSI = F.interpolate(HSI,scale_factor=4, mode='bicubic') ### (b N h w)
        # ca = self.ca_attn(UP_LRHSI)
        UP_LRHSI = UP_LRHSI.clamp_(0, 1)
        # print(np.shape(UP_LRHSI))
        sz = UP_LRHSI.size(2)
        # HSI与MSI的融合
        Data = torch.cat((UP_LRHSI,MSI),1)

        # HSI与MSI的分通道融合
        # Data = torch.zeros(UP_LRHSI.size(0), UP_LRHSI.size(1) + MSI.size(1), UP_LRHSI.size(2), UP_LRHSI.size(2)).cuda()
        # # 将RGB图像分波段融入LR-HSI
        # Data[:, 0:B, :, :] = UP_LRHSI[:, 0:B, :, :]
        # Data[:, B+1, :, :] = MSI[:, 2, :, :]  # B波段
        # Data[:, B+2:G+1, :, :] = UP_LRHSI[:, B+1:G, :, :]
        # Data[:, G+2, :, :] = MSI[:, 1, :, :]  # G波段
        # Data[:, G+3:R+2, :, :] = UP_LRHSI[:, G+1:R, :, :]
        # Data[:, R+3, :, :] = MSI[:, 0, :, :]  # R波段
        # Data[:, R+4:-1, :, :] = UP_LRHSI[:, R+1:-1, :, :]

        # SF = self.SF(UP_LRHSI)
        SF = self.SF(Data)
        # print(np.shape(SF))
        E = rearrange(SF, 'B c H W -> B (H W) c', H = sz)

        # E = rearrange(Data, 'B c H W -> B (H W) c', H = sz)
        # print(np.shape(E))
        # E = self.Embedding(E)
        # print(np.shape(E))

        # 分组式多尺度
        Highpass2, _, _ = self.trans2(E, sz, sz)
        Highpass4, _, _ = self.trans4(E, sz, sz)
        Highpass8, _, _ = self.trans8(E, sz, sz)
        # 单窗口
        # Highpass_win, _, _ = self.trans(E, sz, sz)

        # 级联多尺度 -> 效果很差，不收敛
        # Highpass2 = self.trans8(E, sz, sz) + E
        # Highpass4 = self.trans4(Highpass2, sz, sz) + E
        # Highpass8 = self.trans2(Highpass4, sz, sz) + E

        # Highpass = Highpass8

        # 特征融合
        # Highpass = (Highpass2 + Highpass4 + Highpass8) / 3
        # Highpass = Highpass2 + E
        # Highpass = Highpass4 + E
        # Highpass = Highpass8 + E
        # 单窗口
        # Highpass = Highpass_win + E

        Highpass = (Highpass2 + Highpass4 + Highpass8) / 3 + E
        # print(self.weight_factor)
        # Highpass = self.weight_factor[0] * Highpass2 + self.weight_factor[1] * Highpass4 + self.weight_factor[2] * Highpass8 + E


        Highpass = rearrange(Highpass,'B (H W) C -> B C H W', H = sz)

        Highpass = self.refine(Highpass)


        # 原本的
        # output = Highpass + UP_LRHSI
        # output = output * ca

        # output = Highpass + UP_LRHSI + UP_LRHSI * ca

        # output = Highpass * ca  # 去掉残差学习
        # output = Highpass  + UP_LRHSI  # 去掉通道注意力
        # output = Highpass * ca + UP_LRHSI

        output = Highpass + UP_LRHSI


        output = output.clamp_(0, 1)

        return output, UP_LRHSI, Highpass


if __name__ == '__main__':
    # model = BasicLayer(dim=48, depth=2, num_heads=3, window_size=8, mlp_ratio=1)
    # x = torch.randn(3, 4096, 48)
    # y, _, _ = model(x, 64, 64)
    # print(np.shape(y))
    model = SEST().cuda()
    # model = SEST()
    # print(model)
    LRHSI = torch.randn(3, 31, 16, 16).cuda()
    HRMSI = torch.randn(3, 3, 64, 64).cuda()
    HRHSI = torch.randn(3, 31, 64, 64).cuda()
    output_HRHSI, UP_LRHSI, Highpass = model(LRHSI, HRMSI)
    print('output_HRHSI:{}, UP_LRHSI:{}, Highpass:{}'.format(output_HRHSI.size(), UP_LRHSI.size(), Highpass.size()))
    # print(model.state_dict())
    print('Start training...')
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


