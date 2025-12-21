import torch

def contrastive_search(model, inputs, max_new_token = 100, top_k = 5, alpha = 0.5):
  inputs = inputs.input_ids
  for _ in range(max_new_token):

    with torch.no_grad():
      logits = model(inputs, output_hidden_states = True)

    hi = torch.nn.functional.normalize(logits.hidden_states[-1][0], p = 2, dim = -1) # all hj -> (13, 640)

    # select tok_k
    next_token_logits = logits.logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim = -1)
    top_k_probs, top_k_ids = torch.topk(probs, top_k)

    input_ids = torch.cat((inputs.repeat(top_k, 1), torch.unsqueeze(top_k_ids, dim = -1)), dim = -1) # (5, 14)
    with torch.no_grad():
      outputs = model(input_ids, output_hidden_states = True)

    hv = torch.nn.functional.normalize(outputs.hidden_states[-1][:, -1, :], p = 2, dim = -1) #(5, 640)
    value, ids = torch.max(torch.matmul(hv, hi.T), dim =-1)

    # new inputs (all previous plus current token input)
    inputs = torch.unsqueeze(torch.cat((inputs[0], top_k_ids[torch.argmax((1 - alpha)*top_k_probs - (alpha * value), -1, keepdim = True)])), dim = 0)

  return inputs


