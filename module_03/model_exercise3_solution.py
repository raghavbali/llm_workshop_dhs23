import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config["n_embd"], 3 * config["n_embd"], bias=config["bias"]
        )
        # output projection
        self.c_proj = nn.Linear(config["n_embd"], config["n_embd"], bias=config["bias"])
        # regularization
        self.attn_dropout = nn.Dropout(config["dropout"])
        self.resid_dropout = nn.Dropout(config["dropout"])
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.dropout = config["dropout"]

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config["block_size"], config["block_size"])).view(
                1, 1, config["block_size"], config["block_size"]
            ),
        )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        attn = self.c_attn(x)
        q, k, v = attn.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config["n_embd"], 4 * config["n_embd"], bias=config["bias"]
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config["n_embd"], config["n_embd"], bias=config["bias"]
        )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_proj = nn.Linear(
            config["n_embd"], config["adapter_size"], bias=config["bias"]
        )
        self.nl = nn.ReLU()
        self.up_proj = nn.Linear(
            config["adapter_size"], config["n_embd"], bias=config["bias"]
        )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        x = self.down_proj(x)
        x = self.nl(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config["n_embd"], bias=config["bias"])
        self.mlp = MLP(config)
        if config.get("adapter_size", 0) > 0:
            self.ln_1.requires_grad_(False)
            self.attn.requires_grad_(False)
            self.ln_2.requires_grad_(False)
            self.mlp.requires_grad_(False)
            self.ln_3 = LayerNorm(config["n_embd"], bias=config["bias"])
            self.adapter = Adapter(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        if hasattr(self, "adapter"):
            x = x + self.adapter(self.ln_3(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config["vocab_size"],
                    config["n_embd"],
                    padding_idx=config.get("pad_token", None),
                ),
                wpe=nn.Embedding(config["block_size"], config["n_embd"]),
                drop=nn.Dropout(config["dropout"]),
                h=nn.ModuleList([Block(config) for _ in range(config["n_layer"])]),
                ln_f=LayerNorm(config["n_embd"], bias=config["bias"]),
            )
        )
        if config.get("n_classes", 0) <= 0:
            self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying
        else:
            self.c_head = nn.Linear(config["n_embd"], config["n_classes"], bias=False)

        if config.get("freeze", False):
            self.transformer.requires_grad_(False)

        if config.get("adapter_size", 0) > 0:
            self.transformer.wte.requires_grad_(False)
            self.transformer.wpe.requires_grad_(False)
            self.transformer.ln_f.requires_grad_(False)
            if hasattr(self, "lm_head"):
                self.lm_head.requires_grad_(False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("up_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config["n_layer"])
                )

        # report number of parameters
        print(
            "total number of parameters:",
            self.get_num_params(),
            "learnable:",
            self.get_num_params(only_learnable=True),
        )

    def get_num_params(self, non_embedding=True, only_learnable=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        if only_learnable:
            n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            if non_embedding and self.transformer.wpe.weight.requires_grad:
                n_params -= self.transformer.wpe.weight.numel()
        else:
            n_params = sum(p.numel() for p in self.parameters())
            if non_embedding:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, prompts=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config["block_size"]
        ), f"Cannot forward sequence of length {t}, block size is only {self.config['block_size']}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if hasattr(self, "c_head"):
            logits = self.c_head(x[:, -1, :])
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                loss = F.cross_entropy(logits, targets.view(-1))
            else:
                loss = None
        else:
            if targets is not None:
                # if we are given some desired targets also calculate the loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.config.get("pad_token", -100),
                )
            else:
                # inference-time mini-optimization: only forward the lm_head on the very last position
                logits = self.lm_head(
                    x[:, [-1], :]
                )  # note: using list [-1] to preserve the time dim
                loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        end_token=None,
        prompt=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config["block_size"]
                else idx[:, -self.config["block_size"] :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, prompts=prompt)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next == end_token:
                break

        return idx
