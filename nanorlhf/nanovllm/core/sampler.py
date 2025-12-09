import torch


class Sampler:
    def prepare_sample(self, seqs, eps):
        temperatures = [seq.temperature + eps for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ps = [seq.top_p for seq in seqs]
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ps = top_ps.clamp(min=0.0, max=1.0)
        return {"temperatures": temperatures, "top_ps": top_ps}

    def sample(self, seqs, logits, eps=1e-12):
        sample_params = self.prepare_sample(seqs, eps)
        logits = logits.float() / sample_params['temperatures'].unsqueeze(-1)

        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        cutoff = sample_params['top_ps'].unsqueeze(-1)
        mask = cumulative_probs > cutoff
        mask[..., 0] = False

        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
        probs_sum = probs_sum.clamp_min(eps)
        sorted_probs = sorted_probs / probs_sum

        sampled_sorted_indices = torch.multinomial(sorted_probs, num_samples=1)
        next_tokens_tensor = sorted_indices.gather(-1, sampled_sorted_indices).squeeze(-1)
        next_tokens = next_tokens_tensor.tolist()
        return next_tokens
