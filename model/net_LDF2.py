import os
import warnings
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from einops.layers.torch import Rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath # 导入 DropPath

# from denoising_diffusion_pytorch.simple_diffusion import ResnetBlock, LinearAttention
from model.simple_diffusion import ResnetBlock,LinearAttention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            time_token = x[:, 0, :].reshape(B, 1, C)
            x_ = x[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)  # Fixme: Check Here
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = torch.cat((time_token, x_), dim=1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# 在 model/net.py 文件中的 Block 类定义处

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 use_cache=False):  # 构造时决定是否缓存
        super().__init__()
        self.use_cache = use_cache
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.use_cache:
            self._cached_attn_out = None
            self._cached_mlp_out = None

    def clear_cache(self):
        if self.use_cache:
            self._cached_attn_out = None
            self._cached_mlp_out = None

    # 在 model/net.py 文件中的 Block 类的 forward 方法定义处

    def forward(self, x, H, W, step_counter=None, T=None): # <<< [MODIFIED] 添加 step_counter 和 T 参数
        # 检查是否应该使用缓存
        # Block 现在根据 PVT 传递的 step_counter 和 T 来决定是否使用缓存
        should_use_cache = (
            self.use_cache and
            self._cached_attn_out is not None and
            self._cached_mlp_out is not None and
            (T is None or step_counter is None or step_counter < T) # <<< [NEW] 核心判断条件
        )
        if should_use_cache:
            x = x + self._cached_attn_out
            x = x + self._cached_mlp_out
            print("进行一个继承Block")
            return x
        # print("并不进行一个继承Block")
        attn_out = self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + attn_out

        mlp_out = self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x + mlp_out

        # 检查是否应该保存缓存
        should_save_cache = (
            self.use_cache and
            (T is None or step_counter is None or step_counter < T) # <<< [NEW] 核心判断条件
        )
        if should_save_cache:
            self._cached_attn_out = attn_out.detach()
            self._cached_mlp_out = mlp_out.detach()

        return x

class OverlapPatchEmbed(nn.Module): 
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, mask_chans=0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if mask_chans != 0:
            self.mask_proj = nn.Conv2d(mask_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                       padding=(patch_size[0] // 2, patch_size[1] // 2))
            # set mask_proj weight to 0
            self.mask_proj.weight.data.zero_()
            self.mask_proj.bias.data.zero_()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):# #x是Image mask是xt噪声
        x = self.proj(x)
        # Do a zero conv to get the mask
        if mask is not None:
            mask = self.mask_proj(mask)
            x = x + mask
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def timestep_embedding(timesteps, dim, max_period=10000):  #timesteps是一个Batch长度的 相同的time
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding #（B,C)


class PyramidVisionTransformerImpr(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], mask_chans=1, 
                 cache_T=2, cache_mask=None, block_cache_mask=None,
                 T=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.mask_chans = mask_chans

        # time_embed
        self.time_embed = nn.ModuleList()
        for i in range(len(embed_dims)):
            self.time_embed.append(nn.Sequential(
                nn.Linear(embed_dims[i], 4 * embed_dims[i]),
                nn.SiLU(),
                nn.Linear(4 * embed_dims[i], embed_dims[i]),
            ))

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0], mask_chans=mask_chans)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # === 缓存配置 ===
        if cache_mask is None:
            cache_mask = {'c1': False, 'c2': False, 'c3': False, 'c4': False}
        if block_cache_mask is None:
            block_cache_mask = {'c1': False, 'c2': False, 'c3': False, 'c4': False}

        self.cache_mask = cache_mask
        self.block_cache_mask = block_cache_mask
        self.cache_T = cache_T
        self.cache_counter = 0
        # === 全局开关配置 ===
        self.T = T # <<< [NEW] 设置切换到 full compute 的步数
        self.step_counter = 0 # <<< [NEW] 用于跟踪 forward 调用次数
        # 保存原始的 mask 配置，以便在需要时恢复
        self.original_cache_mask = cache_mask.copy() # <<< [NEW]
        self.original_block_cache_mask = block_cache_mask.copy() # <<< [NEW]        
        # 决定每个 stage 的 Block 是否启用缓存（根据 block_cache_mask）
        use_cache_s1 = block_cache_mask.get('c1', False)
        use_cache_s2 = block_cache_mask.get('c2', False)
        use_cache_s3 = block_cache_mask.get('c3', False)
        use_cache_s4 = block_cache_mask.get('c4', False)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], use_cache=use_cache_s1)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], use_cache=use_cache_s2)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], use_cache=use_cache_s3)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
            attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3], use_cache=use_cache_s4)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # stage-level 缓存存储
        self.cache = {'c1': None, 'c2': None, 'c3': None, 'c4': None}

    def _clear_all_cache(self):
        # 清空 stage 缓存（按 cache_mask）
        for key in self.cache:
            if self.cache_mask.get(key, False):
                self.cache[key] = None

        # 清空 Block 缓存（按 block_cache_mask）
        stages = [
            (self.block_cache_mask.get('c1', False), self.block1),
            (self.block_cache_mask.get('c2', False), self.block2),
            (self.block_cache_mask.get('c3', False), self.block3),
            (self.block_cache_mask.get('c4', False), self.block4),
        ]
        for should_cache, block_list in stages:
            if should_cache:
                for blk in block_list:
                    blk.clear_cache()

    # 在 model/net.py 文件中的 PyramidVisionTransformerImpr 类的 forward 方法定义处

    def forward(self, x, timesteps, cond_img):
        if self.step_counter==0:
            for key in self.cache:
                self.cache[key]=None
        # 检查是否需要禁用缓存
        if self.T is not None and self.step_counter >= self.T:
            # 临时将所有 mask 设置为 False，实现 "full compute"
            self.cache_mask = {k: False for k in self.cache_mask}
            self.block_cache_mask = {k: False for k in self.block_cache_mask}
            # print(f"Step {self.step_counter}: Switched to full compute (T={self.T})")
        else:
            # 恢复原始的 mask 配置
            if self.step_counter < self.T:
                 self.cache_mask = self.original_cache_mask.copy()
                 self.block_cache_mask = self.original_block_cache_mask.copy()

        B = x.shape[0]
        outs = []

        # === Stage 1 ===
        if self.cache_mask.get('c1', False) and self.cache['c1'] is not None:
            print("加载c1")
            c1 = self.cache['c1']
            x = c1
        else:
            # print("计算c1")
            time_token = self.time_embed[0](timestep_embedding(timesteps, self.embed_dims[0])).unsqueeze(1)
            x, H, W = self.patch_embed1(cond_img, x)
            x = torch.cat([time_token, x], dim=1)
            for blk in self.block1:
                # <<< [MODIFIED] 传递 step_counter 和 T 给 Block
                x = blk(x, H, W, step_counter=self.step_counter, T=self.T)
            x = self.norm1(x)
            x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            c1 = x
            if self.cache_mask.get('c1', False):
                self.cache['c1'] = c1.detach()
        outs.append(c1)

        # === Stage 2 ===
        if self.cache_mask.get('c2', False) and self.cache['c2'] is not None:
            print("加载c2")
            c2 = self.cache['c2']
            x = c2
        else:
            # print("计算c2")
            time_token = self.time_embed[1](timestep_embedding(timesteps, self.embed_dims[1])).unsqueeze(1)
            x, H, W = self.patch_embed2(x)
            x = torch.cat([time_token, x], dim=1)
            for blk in self.block2:
                # <<< [MODIFIED] 传递 step_counter 和 T 给 Block
                x = blk(x, H, W, step_counter=self.step_counter, T=self.T)
            x = self.norm2(x)
            x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            c2 = x
            if self.cache_mask.get('c2', False):
                self.cache['c2'] = c2.detach()
        outs.append(c2)

        # === Stage 3 ===
        if self.cache_mask.get('c3', False) and self.cache['c3'] is not None:
            c3 = self.cache['c3']
            print("加载c3")
        else:
            # print("计算c3")
            time_token = self.time_embed[2](timestep_embedding(timesteps, self.embed_dims[2])).unsqueeze(1)
            x, H, W = self.patch_embed3(x)
            x = torch.cat([time_token, x], dim=1)
            for blk in self.block3:
                # <<< [MODIFIED] 传递 step_counter 和 T 给 Block
                x = blk(x, H, W, step_counter=self.step_counter, T=self.T)
            x = self.norm3(x)
            x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            c3 = x
            if self.cache_mask.get('c3', False):
                self.cache['c3'] = c3.detach()
        outs.append(c3)

        # === Stage 4 ===
        if self.cache_mask.get('c4', False) and self.cache['c4'] is not None:
            c4 = self.cache['c4']
            print("加载c4")
        else:
            # print("计算c4")
            time_token = self.time_embed[3](timestep_embedding(timesteps, self.embed_dims[3])).unsqueeze(1)
            x, H, W = self.patch_embed4(c3)
            x = torch.cat([time_token, x], dim=1)
            for blk in self.block4:
                # <<< [MODIFIED] 传递 step_counter 和 T 给 Block
                x = blk(x, H, W, step_counter=self.step_counter, T=self.T)
            x = self.norm4(x)
            x = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            c4 = x
            if self.cache_mask.get('c4', False):
                self.cache['c4'] = c4.detach()
        outs.append(c4)

        # 周期清空
        self.step_counter += 1

        # 周期清空 (逻辑可能需要根据 T 的定义调整)
        if self.step_counter <= self.T:
            self.cache_counter += 1
            if self.cache_counter >= self.cache_T:
                self._clear_all_cache()
                self.cache_counter = 0

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        time_token = x[:, 0, :].reshape(B, 1, C)  # Fixme: Check Here
        x = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([time_token, x], dim=1)
        return x


class pvt_v2_b0(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b1(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b2(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b3(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b4_m(PyramidVisionTransformerImpr):
    def __init__(self, cache_T=2, cache_mask=None, block_cache_mask=None, T=None, **kwargs):  # <<< [MODIFIED] 添加 T 参数
        super(pvt_v2_b4_m, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, cache_T=cache_T, cache_mask=cache_mask, block_cache_mask=block_cache_mask, T=T, **kwargs) # <<< [MODIFIED] 传递 T


class pvt_v2_b4(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


class pvt_v2_b5(PyramidVisionTransformerImpr):
    def __init__(self, **kwargs):
        super(pvt_v2_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)


# Utility functions and Decoder (unchanged, with original structure)
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=factor, p2=factor),
        nn.Conv2d(dim * (factor ** 2), dim if dim_out is None else dim_out, 1)
    )


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = dim if dim_out is None else dim_out
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module): #dim==256
    def __init__(self, dims, dim, class_num=2, mask_chans=1):
        super(Decoder, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim) #LE
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.time_embed_dim = embedding_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, 4 * self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_embed_dim, self.time_embed_dim),
        )

        resnet_block = partial(ResnetBlock, groups=8)
        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=7, padding=3, stride=4,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=self.time_embed_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 8, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )

        self.pred = nn.Sequential(
            # ConvModule(in_channels=embedding_dim//8+1, out_channels=embedding_dim//8, kernel_size=1,
            #            norm_cfg=dict(type='BN', requires_grad=True)),
            nn.Dropout(0.1),
            nn.Conv2d(embedding_dim // 8, self.num_classes, kernel_size=1)
        )

    def forward(self, inputs, timesteps, x):
        t = self.time_embed(timestep_embedding(timesteps, self.time_embed_dim))

        c1, c2, c3, c4 = inputs
        # print("c1:",c1.shape)
        # print("c2:",c2.shape)
        # print("c3:",c3.shape)
        # print("c4:",c4.shape)
        ##############################################
        _x = [x]
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
                _x.append(x)
            else:
                x = blk(x)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
    
        L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))

        # fusion x_feat and x then transposed conv
        x = torch.cat([_c, x], dim=1)
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                x = blk(x, t)
            else:
                x = blk(x)
        # x = self.pred(torch.cat([x, _x.pop(-1)], dim=1))
        x = self.pred(x)
        return x, c1, c2, c3, c4


class net(nn.Module):
    def __init__(self, class_num=1, mask_chans=1,
                 backbone_cache_T=2,
                 backbone_cache_mask=None,
                 backbone_block_cache_mask=None,
                 T=6,
                 **kwargs):
        super(net, self).__init__()
        self.class_num = class_num

        # 默认缓存配置
        if backbone_cache_mask is None:
            backbone_cache_mask = {'c1': False, 'c2': False, 'c3': False, 'c4': False}
        if backbone_block_cache_mask is None:
            backbone_block_cache_mask = {'c1': False, 'c2': False, 'c3': False, 'c4': False}

        # >>>>>>>>>> 双 PVT 主干（完全独立） <<<<<<<<<<
        self.backbone_edge = pvt_v2_b4_m(
            in_chans=3,
            mask_chans=mask_chans,
            cache_T=backbone_cache_T,
            cache_mask=backbone_cache_mask,
            block_cache_mask=backbone_block_cache_mask,
            T=T
        )

        self.backbone_center = pvt_v2_b4_m(
            in_chans=3,
            mask_chans=mask_chans,
            cache_T=backbone_cache_T,
            cache_mask=backbone_cache_mask,
            block_cache_mask=backbone_block_cache_mask,
            T=T
        )

        # >>>>>>>>>> 双 Decoder（参数独立） <<<<<<<<<<
        self.decode_head_edge = Decoder(
            dims=[64, 128, 320, 512],
            dim=256,
            class_num=class_num,
            mask_chans=mask_chans
        )

        self.decode_head_center = Decoder(
            dims=[64, 128, 320, 512],
            dim=256,
            class_num=class_num,
            mask_chans=mask_chans
        )

        # >>>>>>>>>> LDF 风格融合头（简化版 out1） <<<<<<<<<<
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

        self._init_weights()

    def forward(self, x, timesteps, cond_img):
        # ====== Branch 1: Edge-focused ======
        features_edge = self.backbone_edge(x, timesteps, cond_img)
        pred_edge, _, _, _, _ = self.decode_head_edge(features_edge, timesteps, x)

        # ====== Branch 2: Center-focused ======
        features_center = self.backbone_center(x, timesteps, cond_img)
        pred_center, _, _, _, _ = self.decode_head_center(features_center, timesteps, x)

        # ====== LDF 风格融合（生成 out1） ======
        fused = torch.cat([pred_edge, pred_center], dim=1)  # (B, 2, H, W)
        pred_final = self.linear_fuse(fused)                # (B, 1, H, W)
        # print("pred_final:",pred_final.shape)
        # print("pred_edge:",pred_edge.shape)
        # print("pred_center:",pred_center.shape)

        return pred_final, pred_edge, pred_center

    def _download_weights(self, model_name):
        _available_weights = [
            'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3',
            'pvt_v2_b4', 'pvt_v2_b4_m', 'pvt_v2_b5',
        ]
        assert model_name in _available_weights, f'{model_name} is not available now!'
        return "./pretrained_weights/Anonymity/pvt_pretrained/pvt_v2_b4_m.pth"

    def _init_weights(self):
        pretrained_dict = torch.load(self._download_weights('pvt_v2_b4_m'))
        
        # Edge backbone
        model_dict_edge = self.backbone_edge.state_dict()
        pretrained_dict_edge = {k: v for k, v in pretrained_dict.items() if k in model_dict_edge}
        model_dict_edge.update(pretrained_dict_edge)
        self.backbone_edge.load_state_dict(model_dict_edge, strict=False)

        # Center backbone
        model_dict_center = self.backbone_center.state_dict()
        pretrained_dict_center = {k: v for k, v in pretrained_dict.items() if k in model_dict_center}
        model_dict_center.update(pretrained_dict_center)
        self.backbone_center.load_state_dict(model_dict_center, strict=False)

    @torch.inference_mode()
    def sample_unet(self, x, timesteps, cond_img):
        pred_final, _, _ = self.forward(x, timesteps, cond_img)
        return pred_final

    def extract_features(self, cond_img):
        return cond_img

class EmptyObject(object):
    def __init__(self, *args, **kwargs):
        pass

# Required import for ConvModule (assuming mmcv is available)
try:
    from mmcv.cnn import ConvModule
except ImportError:
    # Fallback if mmcv not available (for minimal testing)
    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, norm_cfg=None, **kwargs):
            super().__init__()
            padding = kernel_size // 2
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            if norm_cfg and norm_cfg.get('type') == 'BN':
                self.norm = nn.BatchNorm2d(out_channels)
            else:
                self.norm = nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.act(self.norm(self.conv(x)))