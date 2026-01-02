
import torch
import math
def CECS(model, input, max_new_token = 100, end_token_id = 106, alpha_0 = 0.8, tau_0 = 0.5, beta_0 = 0.2):
  eps = 1e-9
  alpha = alpha_0
  tau_0 = tau_0
  beta_0 = beta_0
  lambda_t = 0.4

  input = input.input_ids
  for _ in range(max_new_token):

    with torch.no_grad():
      out = model(input, output_hidden_states = True)
    
    probs = torch.softmax(out.logits[:, -1, :], dim = -1)
    hi = torch.nn.functional.normalize(out.hidden_states[-1][0], p = 2, dim = -1)

    p = probs[0]
    H = -(p * torch.log(p)).sum() / math.log(p.numel())

    alpha_t = min(0.05, 0.2 * H.item())

    counts = torch.zeros_like(p)
    for tok in input[0]:
        counts[tok] += 1

    # phi(v)
    phi = counts / (counts + 1)

    # adaptive penalty
    penalty = 1 - (alpha_t * phi)
    penalty = penalty.clamp(min=0.0)

    # apply penalty
    p_pen = p * penalty

    p_pen = p_pen / p_pen.sum()

    tt = tau_0 *(1 + alpha * (H.item()))
    p_pen = torch.softmax(torch.log(p_pen)/tt, dim = -1)

    H = -(p_pen * torch.log(p_pen + eps)).sum() / math.log(p_pen.numel())

    beta_t = beta_0 * H
    tau_t = beta_t * p_pen.max()
    
    # candidate indices
    candidates = torch.where(p_pen >= tau_t)[0]

    input_t = torch.cat((input.repeat(len(candidates), 1), torch.unsqueeze(candidates, dim = -1)), dim = -1)

    with torch.no_grad():
      outputs = model(input_t, output_hidden_states = True)

    hv = torch.nn.functional.normalize(outputs.hidden_states[-1][:, -1, :], p = 2, dim = -1)

    value, ids = torch.max(torch.matmul(hv, hi.T), dim = -1)

    token_id = candidates[torch.argmax(lambda_t * torch.log(p_pen[candidates] + eps) - (1- lambda_t) * value)]

    if token_id == end_token_id:
      input = torch.concat((input[0], torch.tensor([token_id])))
      input = torch.unsqueeze(input, dim = 0)
      break

    input = torch.concat((input[0], torch.tensor([token_id])))
    input = torch.unsqueeze(input, dim = 0)

  return input