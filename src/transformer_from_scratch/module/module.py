from transformers import PretrainedConfig

#HuggingFace的类
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
import torch
import math
import torch.nn as nn
from torch.nn import init
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
#继承nn.Module类
class RMSNorm(nn.Module):#基类继承，括号里是父类
    #init初始化
    def __init__(self,dim:int,eps:float=32):
        super.__init__()#进行参数化
        self.dim=dim
        self.eps=eps
        #利用Parameter类，将weight注册进RMSNorm实例的参数列表 `._parameters`，'.parameters'是父类属性
        self.weight=nn.Parameter(torch.ones(dim))
    #_norm下划线代表私用函数,这是RMSNorm公式的主要实现，最后乘上超参数gamma，即这里的weight
    def _norm(self,x):#x是每个token向量的单个通道，之后要对每个token向量都遍历通道
        #transformer中所有量都是张量，这里x是一维数组，其中最后一维为隐藏层维度，其等于1代表x是一维数组
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
        #去掉keepdim=true会导致取平均值之后x的最后一维直接消失
    #forward底层自动调用该函数
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)
        #float()默认是转化成32，因为eps是32位，所以以防万一将x临时转化成32位
        #嵌入层的词向量是词嵌入向量，该嵌入向量在transformer中经过任意变换后都是token向量
def precompute_freqs(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 1. 初始化标准 RoPE 频率。
    # torch.arange(0, dim, 2) 生成 [0, 2, 4, ... dim-2]
    # 计算出的 freqs 就是标准的 1 / (base ** (2i / d)),标准 RoPE 频率 = base=10000 时算出的一系列的 theta_i
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )

    if rope_scaling is not None:
        # 2. 从配置字典中提取 YaRN 的超参数
        # orig_max: 模型预训练时的原始最大长度（例如 Llama-2 是 2048 或 4096）
        # factor: 要扩展的倍数 s (比如从 2k 扩展到 32k，factor 就是 16)，高频是瓶颈，当高频能扩大外推k倍，整体就能扩大k倍
        # beta_fast (对应论文中的 α): 高频边界，波长比例大于此值的维度不缩放
        # beta_slow (对应论文中的 β): 低频边界，波长比例小于此值的维度全量缩放
        # attn_factor: 注意力温度补偿，由于距离拉长导致注意力分布发散（变平缓），需要乘上一个系数让注意力重新“聚焦”
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        #元组结构进行变量赋值

        # 只有当要推断的长度大于原始训练长度时，才应用缩放
        if end / orig_max > 1.0:
            # 3. 使用前文推导的公式，定义波长比例 b 到维度索引 i 的映射函数
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )

            # 4. 计算高频区和低频区的维度切分点
            # low: 不需要缩放的高频部分的最高索引
            # high: 需要完全缩放的低频部分的最低索引
            low, high = (
                max(math.floor(inv_dim(beta_fast)), 0),
                min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1),
            )

            # 5. 计算混合因子 γ (Ramp)
            # 在 low 之前，ramp 为 0；在 high 之后，ramp 为 1；在 low 和 high 之间，线性过渡。
            # clamp 函数限制了数值只能在 [0, 1] 之间。
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )

            # 6. 频率融合公式：f'(i) = f(i) * ((1-γ) + γ/s)
            # 当 ramp=0 时（高频）：系数为 1，保持原频率不变。
            # 当 ramp=1 时（低频）：系数为 1/factor，即对频率进行线性插值缩放。
            # ramp在0-1之间时：平滑过渡。
            freqs = freqs * (1 - ramp + ramp / factor)

    # 7. 根据目标长度 end，生成位置索引向量 t
    t = torch.arange(end, device=freqs.device)

    # 8. 计算外积：将位置 t 与处理好的频率 freqs 相乘，得到每个位置的旋转角度 θ
    freqs = torch.outer(t, freqs).float()

    # 9. 计算 Cos 和 Sin，并应用注意力补偿系数 (attn_factor)，RoPE本质就是将角度对应一个注意力分数logit
    #cis 就是旋转矩阵的复数简写形式
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#增加unsqueeze_dim=1方便进行广播
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
    #这里的x是刚经过RMSNorm和Linear映射的张量，所以会有4个维度
    bs,slen,num_key_value_heads,head_dim=x.shape()
    if n_rep == 1:
        return x 
    return (
        x[:,:,:,None,:]
        .expand(bs,slen,num_key_value_heads,n_rep,head_dim)
        #将5维的x重新变成原来的4维
        .reshape(bs,slen,num_key_value_heads*n_rep,head_dim)
    )

class Attention(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super().__init__()

        #如果没配置k,v头的数量就退化成MHA
        #全局静态属性
        self.num_key_value_heads=(
            args.num_attention_heads 
            if args.num_key_value_heads is None 
            else args.self.num_key_value_heads
        )

        assert args.num_attention_heads%self.num_key_value_heads==0
        "num_attention_heads must be divisible by num_key_value_heads"

        #本地GPU属性，可被动态修改
        self.n_local_heads=args.num_attention_heads
        self.n_local_kv_heads=self.num_key_value_heads
        self.n_rep=self.n_local_heads//self.n_local_kv_heads
        self.head_dim=args.hidden_size//args.num_attention_heads

        #RMSNorm后的Linear投影矩阵
        #Linear只对最后的维度做操作，最后维度一定是hidden_dim，也就是token向量的维度
        #1，Q投影矩阵，变换后维度不变
        self.q_proj=nn.Linear(
            args.hidden_size,args.num_attention_heads*self.head_dim,bias=False
        )
        #2，K投影矩阵，维度降低,Q,K,V的head_dim都是相等的
        self.k_proj=nn.Linear(
            args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False
        )
        #3，V投影矩阵，维度降低
        self.v_proj=nn.Linear(
            args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False
        )
        #4，O输出投影矩阵
        # 对多头注意力拼接之后的多头 V 结果做线性变换，每个头的输出本身就是「V 的加权求和」
        #这里将输出维度调整为原来的hidden_size
        self.o_proj=nn.Linear(
            args.num_attention_heads*self.head_dim,args.hidden_size,bias=False               
        )

        #定义一下接下来可能要用到的变量
        self.attn_dropout=nn.Dropout(args.dropout)
        #残差
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

        def forward(
            self,
            x:torch.Tensor,
            position_embeddings:Tuple[torch.Tensor,torch.Tensor],
            past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
            use_cache=False,
            attention_mask:Optional[torch.Tensor]=None,
        ):
            #在这里对seq_len进行动态赋值，当seq_len>1时，相当于新传入的token向量大于1，则在进行掩码时
            #取倒数seq_len长度的key进行掩码操作
            bsz,seq_len,_=x.shape
            xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
            xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

            cos,sin=position_embeddings
            #对q,k矩阵进行旋转
            xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)

            #kv_cache实现
            #不能在-1维拼接，会改变向量长度，在1维拼接能正好将两个向量拼接在一起
            if past_key_value is not None:
                xk=torch.cat([past_key_value[0],xk],dim=1)
                xv=torch.cat([past_key_value[1],xv],dim=1)
            past_kv=(xk,xv) if use_cache else None

            xq,xk,xv=(
                xq.transpose(1,2),
                repeat_kv(xk,self.n_rep).transpose(1,2),
                repeat_kv(xv,self.n_rep).transpose(1,2),
            )

            if (
                self.flash
                and (seq_len > 1)
                and (past_key_value is None)
                and (attention_mask is None or torch.all(attention_mask == 1))
            ):
                output=F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                )

            else:
                scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
                #这里scores的倒数第一维是key长度，倒数第二维是q长度
                scores[:,:,:,-seq_len:]+=torch.triu(
                    torch.full((seq_len,seq_len),float("-inf"),device=scores.device),
                    #diagonal=0是主对角线
                    diagonal=1
                )

                #padding 掩码 + softmax 求注意力权重 + 用权重加权 V。
                #attention_mask[b, t] = 1：第 b 条样本，第 t 个位置，真实 token，允许看
                #attention_mask[b, t] = 0：第 b 条样本，第 t 个位置，pad填充，禁止看
                #attention_mask is None：batch 所有句子一样长，没有 pad，不需要处理 pad
                if attention_mask is not None:
                    #attention_mask [B, Lk],unsqueeze(1)→[B, 1, Lk],unsqueeze(2)→[B, 1, 1, Lk]
                    extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                    #如果原始 mask=1（真实 token）：`1-1 =0` → 0 * -1e9 = 0，加到 scores 上，分数不变
                    #如果原始 mask=0（padding）：`1-0 =1` → 1 * -1e9 = -1e9
                    #padding 位置的分数会被加上一个超级小的负数。
                    extended_attention_mask=(1.0-extended_attention_mask)*(-1e9)
                    scores+=extended_attention_mask

                scores=F.softmax(scores.float(),dim=-1).type_as(xq)
                scores=self.attn_dropout(scores)
                output=scores@xv

            #H=num_heads，L=seq_len，Dh=head_dim
            #原：[bsz, H, L, Dh]
            #transpose(1,2) → [bsz, L, H, Dh]
            #.reshape(bsz, seq_len, -1)，-1 自动推导：H * Dh。[bsz, L, H, Dh] → [bsz, L, H*Dh]
            output=output.transpose(1,2).reshape(bsz.seq_len,-1)
            #这里没有残差连接，实际上只是再经过了一层dropout
            output=self.resid_dropout(self.o_proj(output))
            return output,past_kv

class FeedForward(nn.Module):
    def __init__(self,config:MokioMindConfig):
        super().__init__()
        #SwiGLU升维系数，为了保持总参数不变，设置为接近三分之八
        if config.intermediate_size is None:
            intermediate_size=int(config.hidden_size*8/3)
            #向下对齐64Tensor Core颗粒度
            config.intermediate_size=64*((intermediate_size+64-1)//64)
            self.gate_proj=nn.Linear(
                config.hidden_size,config.intermediate_size,bias=False
            )
            self.down_proj=nn.Linear(
                config.intermediate_size,config.hidden_size,bias=False
            )
            self.up_proj=nn.Linear(
                config.hidden_size,config.intermediate_size,bias=False
            )
            self.dropout=nn.Dropout(config.dropout)
            #hidden_act是silu字符串
            self.act_fn=ACT2FN[config.hidden_act]

        def forward(self,x):
            gated=self.act_fn(self.gate_proj(x))*self.up_proj(x)
            return self.dropout(self.down_proj(gated))

#MoE部分
class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MoEFeedForward(nn.Module):  # ！修正：原MoEFeedForaward拼写错误
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config) for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config) for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=x.dtype)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else:
                    y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(
                        p.sum() for p in expert.parameters()
                    )
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache
class MokioMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:MokioMindConfig):
        super.__init__()
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=self.hidden_size//self.num_attention_heads
        self.attention=Attention(config)

        self.layer_id=layer_id
        self.input_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(
            config.hidden_size,eps=config.rms_norm_eps
        )
        #前馈网络FFN
        self.mlp=(
            FeedForward(config)
            if not config.use_moe
            else MoEFeedForward(config)
        )

    def forward(
        self,
        #上一层transformer块输出的token向量，也就是这层的输入
        hidden_states,
        #RoPE 旋转位置编码
        position_embeddings:Tuple[torch.Tensor,torch.Tensor],
        past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
        use_cache=False,
        attention_mask:Optional[torch.Tensor]=None,
    ):

        res=hidden_states

        hidden_states,present_key_value=self.self_attention(
            self.input_layernorm(hidden_states),#pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        #以下是两个残差的加法
        hidden_states=res+hidden_states
        hidden_states=hidden_states+self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states,present_key_value

#包含所有transformer层和其他层，不是只有单单一层transformer
class MokioMindModel(nn.Module):
    def __init__(self,config:MokioMindConfig):
        super.__init__()
        self.config=config
        self.vocab_size,self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)
        #transformer block集合
        self.layers=nn.ModuleList(
            [MokioMindBlock(l,config) for l in range(self.num_hidden_layers)]
        )
        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        freqs_cos,freqs_sin=precompute_freqs(
            dim=config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,

        input_ids:Optional[torch.Tensor]=None,
        attention_mask:Optional[torch.Tensor]=None,
        #KVCache推理时保存上一轮每一层的 key、value；增量生成新 token 时不用重新计算整段序列，加速推理
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        #训练阶段一般关掉
        use_cache: bool = False,
        **kwargs,
    ):
        #input_ids:[batch,seq_len]
        batch_size,seq_length=input_ids.shape

        #判断是不是 HuggingFace 那种 Cache 对象
        if hasattr(past_key_values, "layers"):
            past_key_values = None

        #创建一个长度等于模型层数、全部元素为 None 的列表，列表元素是(K,V)
        past_key_values=past_key_values or [None]*len(self.layers)

        #RoPE 的起始位置索引
        start_pos=(
            #推理时，假设已经生成了 N 个 token，新进来 1 个 token，`start_pos=N`，
            # RoPE 只需要取 N 这个位置的 cos/sin，不用重新算全部历史 token 的旋转因子
            #past_key_values是一个列表，元素是(K,V),取第0层的transformer层的k值的shape的第1维，也就是
            #past_key_values当前已经存好的序列长度
            #所有层的kv缓存的序列长度都相等：假设第一轮forward传入序列长度是8，从第0层到
            #最后一层，所有层都会缓存这8个token id
            #每升成一个新token就要跑一次forward
            past_key_values[0][0].shape(1) if   past_key_values is not None else 0
        )

        #Embedding+dropout
        hidden_states=self.dropout(
            self.embed_tokens(input_ids)
        )

        position_embeddings=(
            self.freqs_cos[start_pos:start_pos+seq_length],
            self.freqs_sin[start_pos:start_pos+seq_length],
        )

        #收集新的KV Cache
        #zip：按位置两两配对
        presents=[]
        for layer_idx,(layer,past_key_value) in enumerate(
            zip(self.layers,past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        #全部 Transformer 层跑完，最后做 RMS 归一化
        hidden_states=self.norm(hidden_states)

        #MoE混合专家的辅助损失
        aux_loss = sum(
            [
                layer.mlp.aux_loss
                for layer in self.layers
                if isinstance(
                    layer.mlp, MoEFeedForward
                )  # ！修正：原MoEFeedForaward拼写错误
            ],
        hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss

#- 继承 `PreTrainedModel`：能直接用 `.from_pretrained()` 加载权重、保存 checkpoint
#- 继承 `GenerationMixin`：自带 `.generate()` 文本生成函数（写对话、续写不用自己写解码循环）
class MokioMindForCausalLM(PreTrainedModel,GenerationMixin):
    #HF约定变量名
    config_class=MokioMindConfig

    #初始化
    def __init__(self,config:MokioMindConfig):
        super.__init__(config)
        self.model=MokioMindModel(config)
        #语言模型输出头，将transformer输出的向量映射到vocab_size词表维度
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)

        #词嵌入层（embed_tokens） 和 最后的输出投影 lm_head 共用同一套权重。
        # Llama、GPT 很多模型都这么做，减少参数量，提升效果。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        #和 input_ids 同 shape，是目标 token；推理生成时不传 labels
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        hidden_states,past_key_values,aux_loss=self.model(
            #外部函数调用forward，forward把收到的input_ids，原样传给主干MokioMindModel
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )      
        slice_indices=(
            #切片倒数第logits_to_keep个隐藏向量
            slice(-logits_to_keep,None)
            if isinstance(logits_to_keep,int)
            else logits_to_keep
        )
        #给lm_head传入logits_to_keep个隐藏向量,对hidden_states的seq_len维的长度进行切片
        logits=self.lm_head(hidden_states[:,slice_indices,:])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                #view的-1，压扁成一维，并且自动算出维度大小
                #shift_logits.size的-1，取出最后一个维度的大小，即词表大小
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )
        output.aux_loss = aux_loss
        return output