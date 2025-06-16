# LLM Parameters

## Temperature

The sampling temperature to use for text generation. The higher the temperature value is, the less deterministic the output text will be. It is not recommended to modify both temperature and top_p in the same call.

## Top P

The top-p sampling mass used for text generation. The top-p value determines the probability mass that is sampled at sampling time. For example, if top_p = 0.2, only the most likely tokens (summing to 0.2 cumulative probability) will be sampled. It is not recommended to modify both temperature and top_p in the same call.

## Frequency Penalty

Indicates how much to penalize new tokens based on their existing frequency in the text so far, decreasing model likelihood to repeat the same line verbatim.

## Presence Penalty

Positive values penalize new tokens based on whether they appear in the text so far, increasing model likelihood to talk about new topics.
