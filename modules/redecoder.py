import torch
from modules.wavenet import WN
#
class Redecoder(torch.nn.Module):
    def __init__(self, args):
        super(Redecoder, self).__init__()
        self.n_p_codebooks = args.n_p_codebooks # number of prosody codebooks
        self.n_c_codebooks = args.n_c_codebooks # number of content codebooks
        self.codebook_size = 1024 # codebook size
        self.encoder_type = args.encoder_type
        if args.encoder_type == "wavenet":
            self.embed_dim = args.wavenet_embed_dim
            self.encoder = WN(hidden_channels=self.embed_dim, kernel_size=5, dilation_rate=1, n_layers=16, gin_channels=1024
                              , p_dropout=0.2, causal=args.decoder_causal)
            self.conv_out = torch.nn.Conv1d(self.embed_dim, 1024, 1)
            self.prosody_embed = torch.nn.ModuleList(
                [torch.nn.Embedding(self.codebook_size, self.embed_dim) for _ in range(self.n_p_codebooks)])
            self.content_embed = torch.nn.ModuleList(
                [torch.nn.Embedding(self.codebook_size, self.embed_dim) for _ in range(self.n_c_codebooks)])
        elif args.encoder_type == "mamba":
            from modules.mamba import Mambo
            self.embed_dim = args.mamba_embed_dim
            self.encoder = Mambo(d_model=self.embed_dim, n_layer=24, vocab_size=1024,
                                 prob_random_mask_prosody=args.prob_random_mask_prosody,
                                 prob_random_mask_content=args.prob_random_mask_content,)
            self.conv_out = torch.nn.Linear(self.embed_dim, 1024)
            self.forward = self.forward_v2
            self.prosody_embed = torch.nn.ModuleList(
                [torch.nn.Embedding(self.codebook_size, self.embed_dim) for _ in range(self.n_p_codebooks)])
            self.content_embed = torch.nn.ModuleList(
                [torch.nn.Embedding(self.codebook_size, self.embed_dim) for _ in range(self.n_c_codebooks)])
        else:
            raise NotImplementedError

    def forward(self, p_code, c_code, timbre_vec, use_p_code=True, use_c_code=True, n_c=2):
        B, _, T = p_code.size()
        p_embed = torch.zeros(B, T, self.embed_dim).to(p_code.device)
        c_embed = torch.zeros(B, T, self.embed_dim).to(c_code.device)
        if use_p_code:
            for i in range(self.n_p_codebooks):
                p_embed += self.prosody_embed[i](p_code[:, i, :])
        if use_c_code:
            for i in range(n_c):
                c_embed += self.content_embed[i](c_code[:, i, :])
        x = p_embed + c_embed
        x = self.encoder(x.transpose(1, 2), x_mask=torch.ones(B, 1, T).to(p_code.device), g=timbre_vec.unsqueeze(2))
        x = self.conv_out(x)
        return x
    def forward_v2(self, p_code, c_code, timbre_vec, use_p_code=True, use_c_code=True, n_c=2):
        x = self.encoder(torch.cat([p_code, c_code], dim=1), timbre_vec)
        x = self.conv_out(x).transpose(1, 2)
        return x
    @torch.no_grad()
    def generate(self, prompt_ids, input_ids, prompt_context, timbre, use_p_code=True, use_c_code=True, n_c=2):
        from modules.mamba import InferenceParams
        assert self.encoder_type == "mamba"
        inference_params = InferenceParams(max_seqlen=8192, max_batch_size=1)
        # run once with prompt to initialize memory first
        prompt_out = self.encoder(prompt_ids, prompt_context, timbre, inference_params=inference_params)
        for i in range(input_ids.size(-1)):
            input_id = input_ids[..., i]
            prompt_out = self.encoder(input_id, prompt_out, timbre, inference_params=inference_params)

