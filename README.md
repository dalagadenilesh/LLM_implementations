
# Decosing Strategies in LLM

Large Language Model for natural language processing has only ability to predict next word for given prefix. These models are trained to predict next token on massive corpora. Currently LLM are more efficient to solve tasks like text generation, summarization and translation.

Except Vision based LLM, all transformers now a days are Transformer based architecture. Hence Transformer and LLMs are synonymously used. (Note some of the LLMs was there which are based on Recurrent neural network). Since Transformer decoders works as autoregressive for inference, it produces logits from last hidden stage for every token prediction over a fixed vocabulary. In order to generate next word a sequence we need a decoder in order to derive how the generated token sequence is derived from these logits. Hence decoding methods act as bridge between next-token predictors and generators, plays an integral role in transforming LLM into task solver.It has been found that, optimal decoding methods depends on task in hand, model and priority.

Various decoding methods are available and brodly classified into two types.
### deterministic
Deterministic decoding methods are  - Greddy search, Beam Search, Constractive search and context enhanced constractive search
### stichastic ( sampling based)
Temperature scaling, Top-p, top-k


We will be talk little bit about performance of model before jumping into decoding strategies. for a text generation, the optimal text generation is defined by term Coherance, Diversity and fluency

coherance -  how well gneerated text fits together logically and contextually over time
Diversity - How different the generated tokens, phrases or outputs are from each other
semantic quality - how meaningful, informative and non-trivial the content is

Highly coherant is somewhat boring, repatative somewhat and highly Diverse text generation is like story and non informative, nonsense.


Deterministic decoding methods are  - Greddy search, Beam Search, Constractive search and context enhanced constractive search

Deterministice Decoding methods are higly coherant but repetative ones. With Greedy search and Beam Search, we choose tokens with high probability but no control on repetative tokens since we are choosing high probability.

With Beam Search, maximizing cumulative probability for all steps, more likelihood of selecting repetative tokens. this method is higly coherant and very less diverse in nature. As per study, it produces nearly identical beams which not only compuational expensive but also lack of diversity and more generic solution. It seems wasteful of computationa for little gain in performance of text generation quality.

Diverse Beam Search decode to list a diverse sequences that can be used alternative to Beam Search. Divide k most probable sequences into G groups and incorporate diversity term to maximize inter group diversity.

## Constractive search (CS)

Constractive search (CS) is proposed to obtain tokens with high coherant as well as high Diversity. The underlying concept is implemented in context_enhanced_CS.py.

By select top-k token at given time, calculate hidden state of decoder for these tokens. Logically, These tokens are somewhat should have semantic with previous tokens or prefix and instead of choosing randomly, tokens that descriminate enough with respect to previous context or prefix are selected. This can avoid model degeration. 

By concatenating these tokens with prefix, we generate hidden state from model for these tokens and calculate cosine similarity with all tokens hidden state in context. 

Larger degeneration penalty of these tokens menas more similar to context, hence more likely leading to model degenration.

Lets top-k = 3

hv1 - cosine similarity of hv1 with all previous context and max of these,
hv2 - cosine similarity of hv2 with all precious context and max of these,
hv3 - cosine similarity of hv3 with all precious context and max of these

    token at time t = argmax((1 - alpha) * top-k probbility - alpha([hv1, hv2, hv3]))

alpha - range(0, 1) - default should be 0.6.
With alpha = 0, this argmax will be exactly behave like greedy. So alpha will be hyperparameter which impact on the importance of penalty.

With increase in alpha, we are generating more diverse text.


### Context-Enhanced Constrastive search:

In CS, we are constracting high probability and low probability tokens within same distributions to avoid limitations of conventional search method such as greedy and Beam search. in CECS, we are 


We already discussed temperature scaling, temperature will change skewness of probability distribution, introduces more confidence or more uncertainty in text generation. Less skewed probability distribution, lesser the entropy of token distribution, more confident model and vice versa. 

At Every decoding step t, temperature is calculated dynamicaly from entropy, indicates that, temp is calculated based on complexity of generated text so far.

    touT = tou0 * (1 + alpha * fcontext(y1:t-1))

    alpha - scaling factor

As expecetd, tout will increase with increase in entropy.

Instead of penalizing tokens based on cosine similarity with context tokens, CECS implements adaptive penalty mechanism which adjust probability distribution of token as determined bu their occurence in present context.

Instead of penalizing tokens based on cosine similarity with context tokens, CECS implements adaptive penalty mechanism which adjust probability distribution of token as determined bu their occurence in present context.

    lambdaT(yt) = 1 - alphaT * fpenaty(yt, y1:t-1)

fpenalty = penalty for token yt based on prior y1:t-1 context. Counting tokens

    fpenaty(yt, y1:t-1) = count(yt, y1:t-1)/t

count(yt, y1:t-1)  number of times yt has appreared in context y1:t-1 and t is current decosing step.

alphaT is penalty factor, controls degree of penalty at current step. alphaT is context-sensitive and will change based on complexity of current context and model level confidence.

alphaT = alpha0 * Hcontext(y1:t-1)
More spreaded probability distribution indicates more uncertainlty, alohaT increases and strong penalty applied to prevent degenrate or repetative text generation.

    ppen(yt|y1:t-1, x) = p(yt|y1:t-1, x) * lanbdat(yt)

ppen - penealized probability distribution

Lastly, selecting candidate token Ct form penalized probabilty distribution.
Ct = ppen(yt:y1:t-1, x) > beta_t * ppen(yt:y1:t-1, x)
s(yt) = lambda * ppen(yt|y1:t-1) - diversity(candidates)

From above, calculation of s(yt) is similar to CS except ppen are filtere on candidates selection.

