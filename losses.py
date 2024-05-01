import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram


def adversarial_g_loss(y_disc_gen):
    """Hinge loss"""
    loss = 0.0
    for i in range(len(y_disc_gen)):
        stft_loss = F.relu(1 - y_disc_gen[i]).mean().squeeze()
        loss += stft_loss
    return loss / len(y_disc_gen)


def feature_loss(fmap_r, fmap_gen):
    loss = 0.0
    for i in range(len(fmap_r)):
        for j in range(len(fmap_r[i])):
            stft_loss = ((fmap_r[i][j] - fmap_gen[i][j]).abs() /
                         (fmap_r[i][j].abs().mean())).mean()
            loss += stft_loss
    return loss / (len(fmap_r) * len(fmap_r[0]))


def sim_loss(y_disc_r, y_disc_gen):
    loss = 0.0
    for i in range(len(y_disc_r)):
        loss += F.mse_loss(y_disc_r[i], y_disc_gen[i])
    return loss / len(y_disc_r)

# def sisnr_loss(x, s, eps=1e-8):
    # """
    # calculate training loss
    # input:
          # x: separated signal, N x S tensor, estimate value
          # s: reference signal, N x S tensor, True value
    # Return:
          # sisnr: N tensor
    # """
    # if x.shape != s.shape:
        # if x.shape[-1] > s.shape[-1]:
            # x = x[:, :s.shape[-1]]
        # else:
            # s = s[:, :x.shape[-1]]
    # def l2norm(mat, keepdim=False):
        # return torch.norm(mat, dim=-1, keepdim=keepdim)
    # if x.shape != s.shape:
        # raise RuntimeError(
            # "Dimention mismatch when calculate si-snr, {} vs {}".format(
                # x.shape, s.shape))
    # x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    # s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    # t = torch.sum(
        # x_zm * s_zm, dim=-1,
        # keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    # loss = -20. * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    # return torch.sum(loss) / x.shape[0]

LAMBDA_WAV = 100
LAMBDA_ADV = 1
LAMBDA_REC = 1
LAMBDA_COM = 1000
LAMBDA_FEAT = 1
discriminator_iter_start = 2000
def reconstruction_loss(x, G_x, eps=1e-7):
    # NOTE (lsx): hard-coded now
    L = LAMBDA_WAV * F.mse_loss(x, G_x)  # wav L1 loss
    # loss_sisnr = sisnr_loss(G_x, x) #
    # L += 0.01*loss_sisnr
    # 2^6=64 -> 2^10=1024
    # NOTE (lsx): add 2^11
    for i in range(6, 12):
        # for i in range(5, 12): # Encodec setting
        s = 2**i
        melspec = MelSpectrogram(
            sample_rate=16000,
            n_fft=max(s, 512),
            win_length=s,
            hop_length=s // 4,
            n_mels=64,
            wkwargs={"device": G_x.device}).to(G_x.device)
        S_x = melspec(x)
        S_G_x = melspec(G_x)
        l1_loss = (S_x - S_G_x).abs().mean()
        l2_loss = (((torch.log(S_x.abs() + eps) - torch.log(S_G_x.abs() + eps))**2).mean(dim=-2)**0.5).mean()

        alpha = (s / 2) ** 0.5
        L += (l1_loss + alpha * l2_loss)
    return L


def criterion_d(y_disc_r, y_disc_gen, fmap_r_det, fmap_gen_det, y_df_hat_r,
                y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g,
                fmap_s_r, fmap_s_g):
    """Hinge Loss"""
    loss = 0.0
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    for i in range(len(y_disc_r)):
        loss1 += F.relu(1 - y_disc_r[i]).mean() + F.relu(1 + y_disc_gen[
            i]).mean()
    for i in range(len(y_df_hat_r)):
        loss2 += F.relu(1 - y_df_hat_r[i]).mean() + F.relu(1 + y_df_hat_g[
            i]).mean()
    for i in range(len(y_ds_hat_r)):
        loss3 += F.relu(1 - y_ds_hat_r[i]).mean() + F.relu(1 + y_ds_hat_g[
            i]).mean()

    loss = (loss1 / len(y_disc_gen) + loss2 / len(y_df_hat_r) + loss3 /
            len(y_ds_hat_r)) / 3.0

    return loss


def criterion_g(commit_loss, x, G_x, fmap_r, fmap_gen, y_disc_r, y_disc_gen,
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r,
                y_ds_hat_g, fmap_s_r, fmap_s_g, args):
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    feat_loss = (feature_loss(fmap_r, fmap_gen) + sim_loss(
        y_disc_r, y_disc_gen) + feature_loss(fmap_f_r, fmap_f_g) + sim_loss(
            y_df_hat_r, y_df_hat_g) + feature_loss(fmap_s_r, fmap_s_g) +
                 sim_loss(y_ds_hat_r, y_ds_hat_g)) / 3.0
    rec_loss = reconstruction_loss(x.contiguous(), G_x.contiguous(), args)
    total_loss = args.LAMBDA_COM * commit_loss + args.LAMBDA_ADV * adv_g_loss + args.LAMBDA_FEAT * feat_loss + args.LAMBDA_REC * rec_loss
    return total_loss, adv_g_loss, feat_loss, rec_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def adopt_dis_weight(weight, global_step, threshold=0, value=0.):
    # 0,3,6,9,13....这些时间步，不更新dis
    if global_step % 3 == 0:
        weight = value
    return weight


def calculate_adaptive_weight(nll_loss, g_loss, last_layer, args):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(
            nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        print('last_layer cannot be none')
        assert 1 == 2
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 1.0, 1.0).detach()
    d_weight = d_weight * args.LAMBDA_ADV
    return d_weight

def loss_g(codebook_loss,
           inputs,
           reconstructions,
           fmap_r,
           fmap_gen,
           y_disc_r,
           y_disc_gen,
           global_step,
           y_df_hat_r,
           y_df_hat_g,
           y_ds_hat_r,
           y_ds_hat_g,
           fmap_f_r,
           fmap_f_g,
           fmap_s_r,
           fmap_s_g,
           last_layer=None,
           is_training=True,
           args=None):
    """
    args:
        codebook_loss: commit loss.
        inputs: ground-truth wav.
        reconstructions: reconstructed wav.
        fmap_r: real stft-D feature map.
        fmap_gen: fake stft-D feature map.
        y_disc_r: real stft-D logits.
        y_disc_gen: fake stft-D logits.
        global_step: global training step.
        y_df_hat_r: real MPD logits.
        y_df_hat_g: fake MPD logits.
        y_ds_hat_r: real MSD logits.
        y_ds_hat_g: fake MSD logits.
        fmap_f_r: real MPD feature map.
        fmap_f_g: fake MPD feature map.
        fmap_s_r: real MSD feature map.
        fmap_s_g: fake MSD feature map.
    """
    rec_loss = reconstruction_loss(inputs.contiguous(),
                                   reconstructions.contiguous())
    adv_g_loss = adversarial_g_loss(y_disc_gen)
    adv_mpd_loss = adversarial_g_loss(y_df_hat_g)
    adv_msd_loss = adversarial_g_loss(y_ds_hat_g)
    adv_loss = (adv_g_loss + adv_mpd_loss + adv_msd_loss
                ) / 3.0  # NOTE(lsx): need to divide by 3?
    feat_loss = feature_loss(
        fmap_r,
        fmap_gen)  #+ sim_loss(y_disc_r, y_disc_gen) # NOTE(lsx): need logits?
    feat_loss_mpd = feature_loss(fmap_f_r,
                                 fmap_f_g)  #+ sim_loss(y_df_hat_r, y_df_hat_g)
    feat_loss_msd = feature_loss(fmap_s_r,
                                 fmap_s_g)  #+ sim_loss(y_ds_hat_r, y_ds_hat_g)
    feat_loss_tot = (feat_loss + feat_loss_mpd + feat_loss_msd) / 3.0
    d_weight = torch.tensor(1.0)
    # try:
    #     d_weight = calculate_adaptive_weight(rec_loss, adv_g_loss, last_layer, args) # 动态调整重构损失和对抗损失
    # except RuntimeError:
    #     assert not is_training
    #     d_weight = torch.tensor(0.0)
    disc_factor = adopt_weight(
        LAMBDA_ADV, global_step, threshold=discriminator_iter_start)
    if disc_factor == 0.:
        fm_loss_wt = 0
    else:
        fm_loss_wt = LAMBDA_FEAT
    #feat_factor = adopt_weight(args.LAMBDA_FEAT, global_step, threshold=args.discriminator_iter_start)
    loss = rec_loss + d_weight * disc_factor * adv_loss + \
           fm_loss_wt * feat_loss_tot + LAMBDA_COM * codebook_loss.mean()
    return loss, rec_loss, adv_loss, feat_loss_tot, d_weight


def loss_dis(y_disc_r_det, y_disc_gen_det, fmap_r_det, fmap_gen_det, y_df_hat_r,
             y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r,
             fmap_s_g, global_step, args):
    disc_factor = adopt_weight(
        args.LAMBDA_ADV, global_step, threshold=args.discriminator_iter_start)
    d_loss = disc_factor * criterion_d(y_disc_r_det, y_disc_gen_det, fmap_r_det,
                                       fmap_gen_det, y_df_hat_r, y_df_hat_g,
                                       fmap_f_r, fmap_f_g, y_ds_hat_r,
                                       y_ds_hat_g, fmap_s_r, fmap_s_g)
    return d_loss

class AttentionCTCLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(AttentionCTCLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = torch.nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0, 0, 0, 0, 0, 0, 0),
            value=self.blank_logprob)
        cost_total = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid]+1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                :query_lens[bid], :, :key_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            ctc_cost = self.CTCLoss(curr_logprob, target_seq,
                                    input_lengths=query_lens[bid:bid+1],
                                    target_lengths=key_lens[bid:bid+1])
            cost_total += ctc_cost
        cost = cost_total/attn_logprob.shape[0]
        return cost


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
