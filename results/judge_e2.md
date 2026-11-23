| tag | setting | params | lr | seeds | seeds at 99% | two-chain 5-8, seeds at 99% | two-chain 5-8, all seeds | median first step at 99% |
|---|---|---|---|---|---|---|---|---|
| e2 | tied/square | 46,965 | 0.001 | 10 | 5 | 100.0 | 92.9 | 4000 |
| e2fix | tied/square/attn8 | 53,541 | 0.001 | 6 | 5 | 99.9 | 99.2 | 4000 |
| e2 | delta L2 d64 | 106,800 | 0.001 | 2 | 2 | 87.8 | 87.8 | 14000 |
| e2 | delta L4 d64 | 208,312 | 0.001 | 2 | 2 | 64.4 | 64.4 | 10000 |
| e2 | delta L8 d64 | 411,336 | 0.003 | 1 | 1 | 64.4 | 64.4 | 22000 |
| e2 | lstm L2 h128 | 239,144 | 0.003 | 1 | 0 |  | 43.6 |  |
| e2 | tf L12 d512 h8 | 37,909,544 | 0.0001 | 2 | 1 | 76.9 | 64.2 | 26000 |
| e2 | tf L4 d64 h4 | 210,088 | 0.001 | 2 | 0 |  | 45.3 |  |

e2 tied/square: two-chain 5-8 100.0% against the best standard model that reaches 99%, delta L2 d64 at 87.8%: difference +12.2 points (less than the 20 points of the first benefit condition).
e2fix tied/square/attn8: two-chain 5-8 99.9% against the best standard model that reaches 99%, delta L2 d64 at 87.8%: difference +12.0 points (less than the 20 points of the first benefit condition).
