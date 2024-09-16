import torch
from pyae import ArithmeticEncoding

emb = torch.tensor([[0.1, 0.2, 0.5], [0.4, 1.2, 0.8], [3.2, 0.1, 0.5]])

total = emb[:, 0] + emb[:, 1] + emb[:, 2]
mean = total / emb.shape[1]
mean_mat = mean.repeat(3,1)
dist = torch.subtract(emb, mean_mat)
counts = (-dist).exp()
prob = counts / counts.sum(0, keepdim=True)
probs = {}
sample = torch.tensor([0.2, 0.2, 0.5])
sample_dist = torch.subtract(sample, mean)
sample_counts = (-sample_dist).exp()
print(sample_counts)
for i in range(3):
    probs[str(i)] = prob[i, 0].item()

ae = ArithmeticEncoding(frequency_table=probs, save_stages=True)
print(ae.probability_table)
msg = "1"
encoded_msg, encoder , interval_min_value, interval_max_value = ae.encode(msg=msg,


                                                                          probability_table=ae.probability_table)
print(encoded_msg)
decoded_msg, decoder = ae.decode(encoded_msg=0.56,
                                 msg_length=len(msg),
                                 probability_table=ae.probability_table)
print(decoded_msg)
