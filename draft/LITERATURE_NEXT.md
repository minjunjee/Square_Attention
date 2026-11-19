# 다음 단계를 고르려고 한 문헌 조사

2026년 11월 19일 밤에 했다. 조사는 별도 agent에게 맡겼다. agent는 논문마다 arXiv 초록 페이지나 model card를 열어 제목, 첫 저자, 연도, 번호를 확인했다고 보고했다. 나는 그 보고만 읽었고 논문을 직접 읽지 않았다. "초록만"이라고 적은 논문은 agent도 초록만 읽은 것이다. 이 글의 논문 내용은 모두 그 보고에서 옮겼다.

조사한 질문은 네 가지다. 이 조사로 고른 실험은 `SERVER_RESULTS.md` 4.9절의 실험 9이다.

## 1. state를 거듭 곱해서 읽는 방법이 이미 있는가

delta rule로 쓴 state S를 r번 제곱해서 S의 2^r 제곱으로 읽는 방법과 같은 것은 찾지 못했다. agent는 2026년 preprint까지 넣어 열두 번쯤 검색했다. 찾지 못했다는 것이 없다는 증명은 아니다. 가까운 연구는 다음과 같다.

| 논문 | 우리와 겹치는 점과 다른 점 |
|---|---|
| Higher-order Linear Attention (Zhang, 2025, arXiv:2510.27258) | 가장 가깝다. attention의 유사도 행렬을 두 번 적용한 값을 크기가 고정된 state로 계산한다. 알고리즘과 이론만 있고 실험은 없다. state 행렬을 제곱하지 않고, 여러 칸을 따라가는 문제도 다루지 않는다 |
| Learning Associative Inference Using Fast Weight Memory (Schlag, 2020, arXiv:2011.07831) | delta rule과 비슷한 방식으로 쓴 state를 세 번 차례로 읽는다. 읽은 값이 다음 읽기의 key가 된다. state를 제곱하지 않으므로 따라가는 칸 수가 읽는 횟수에 비례한다. 우리의 "순서대로 곱하기" 조건과 같은 방식이다 |
| DeltaProduct (Siems, 2025, arXiv:2502.10297) | token마다 delta rule 갱신을 여러 번 한다. 쓰는 쪽을 바꾼 것이고 읽기는 한 번이다 |
| PaTH Attention (Yang, 2025, arXiv:2505.16381), Log-Linear Attention (Guo, 2025, arXiv:2506.04761) | 초록만. state를 갱신하거나 배치하는 방식을 바꾼 것이고 읽기는 그대로다 |
| Transformers, parallel computation, and logarithmic depth (Sanford, 2024, arXiv:2402.09268) | k칸을 따라가는 문제에 Transformer는 log2 k에 비례하는 깊이면 되지만, 여러 층 RNN은 깊이가 k 이상이거나 폭이 매우 커야 한다는 이론이다. Mamba도 RNN 쪽에 든다. layer 하나 안에서 칸 수를 두 배씩 늘리는 우리 방법의 이론적 근거가 된다. state의 폭이 한계를 정한다는 경고이기도 하다 |
| Transformers Provably Learn Algorithmic Solutions for Graph Connectivity, But Only with the Right Data (Ye, 2025, arXiv:2510.19753) | L층 Transformer가 층을 지나며 인접 행렬의 거듭제곱을 계산한다. 행렬 거듭제곱을 층 수로 얻는 것이고, 우리는 layer 하나의 state에서 얻는다 |
| Loop, Think, & Generalize (Kohli, 2026, arXiv:2604.07822) | 초록만. 같은 block을 반복하는 Transformer를 5칸까지 학습하면, 시험 때 반복을 늘려 10칸까지 푼다. 너무 많이 반복하면 정확도가 떨어진다. 우리 실험 3의 비교군과 같은 계열이다 |
| MesaNet (von Oswald, 2025, arXiv:2506.05233), Don't Read Everything (Le, 2026, arXiv:2606.01294) | 초록만. 읽는 쪽을 바꾼 연구지만 연관을 이어 붙이지는 않는다 |

## 2. key가 많을 때 학습이 어려운 이유와 알려진 처방

이름 수와 state 크기의 관계를 이 변수 binding 문제에서 조사한 연구는 찾지 못했다. code를 무작위로 두고 고정하는 처방을 보고한 연구도 찾지 못했다. 관련된 결과는 다음과 같다.

| 논문 | 우리에게 주는 것 |
|---|---|
| How Do Transformers Learn Variable Binding in Symbolic Programs? (Wu, 2025, arXiv:2505.20896) | 우리 문제의 출처다. 이름 26개, 16줄, 12층 Transformer. 학습이 세 단계를 지난다. 찍기 수준인 약 12%, 첫 줄이나 둘째 줄의 숫자를 답하는 요령 단계인 약 56%, 체인을 따라가는 단계인 99.9% 이상이다. 체인을 따라가는 풀이는 요령 위에 쌓이고 요령은 남는다. 문제 생성기는 가장 긴 체인의 끝을 답하는 요령을 막도록 만들었다. 학습은 프로그램 전체에 대한 다음 token 예측이고 질문은 하나다. seed 6개가 모두 요령을 벗어났다. 이름 수, 학습 순서, loss를 주는 위치를 바꾼 비교는 없다 |
| 위의 Ye 2025 | 모델이 풀 수 있는 크기를 넘는 문제를 학습에 넣으면 요령을 배우고, 풀 수 있는 크기의 문제만 넣으면 행렬 거듭제곱 풀이를 배운다 |
| Learning Compositional Functions with Transformers from Easy-to-Hard Data (Wang, 2025, arXiv:2505.23683) | 함수를 k번 합성하는 문제는 어려운 문제만으로 학습하면 필요한 계산량이 k에 지수적으로 늘어난다는 하한이 있다. 쉬운 문제에서 어려운 문제로 가는 학습 순서, 또는 쉬운 문제와 어려운 문제를 섞은 학습은 다항 시간에 배운다 |
| Revisiting associative recall in modern recurrent models (Okpekpe, 2025, arXiv:2508.19029) | key와 value를 여러 쌍 보여 주고 key를 물어 value를 답하게 하는 MQAR 과제에서, Mamba와 Hyena는 좁은 learning rate 범위에서만 학습에 성공한다. learning rate를 성기게 고르면 못 배운다고 잘못 결론짓게 된다. DeltaNet은 Transformer만큼 learning rate에 둔감했다 |
| Mimetic Initialization Helps State Space Models Learn to Recall (Trockman, 2024, arXiv:2410.11135) | 초록만. 기억해 내기에 실패하는 원인의 일부는 용량이 아니라 최적화이고, 구조를 준 초기화로 고쳐진다 |
| Zoology (Arora, 2023, arXiv:2312.04927) | MQAR는 한 sequence에 질문이 여러 개다. attention은 차원 64로 풀고, gated convolution은 차원이 sequence 길이 이상이어야 한다 |
| How Far Can Transformers Reason? (Abbe, 2024, arXiv:2406.06467) | 초록만. 긴 합성 체인은 끝 답만으로는 배우기 어렵고, 중간 단계에 정답을 주면 배운다 |

## 3. 지금의 hybrid 언어 모델은 delta rule layer를 어떻게 쓰는가

| 모델 | key와 value를 만드는 방식 |
|---|---|
| Gated Delta Networks (Yang, 2024, arXiv:2412.06464) | query, key, value를 서로 다른 linear와 짧은 convolution으로 만든다. query와 key는 길이 1로 맞춘다. state에 decay를 곱한다 |
| Kimi Linear (Kimi Team, 2025, arXiv:2510.26692) | 좌표마다 다른 decay를 쓴다. head마다 key와 value가 128차원이다. Kimi Delta Attention layer 3개에 full attention layer 1개를 섞는다 |
| Qwen3-Next-80B-A3B model card | Gated DeltaNet layer 3개에 attention layer 1개. key 쪽 head 16개, value 쪽 head 32개라 state가 정사각형이 아니다 |
| RWKV-7 (Peng, 2025, arXiv:2503.14456) | head마다 64×64 state. 지울 key와 새로 쓸 key를 따로 둔다 |
| Gated DeltaNet-2 (Hatamizadeh, 2026, arXiv:2605.22791) | 초록만. 지우는 gate와 쓰는 gate를 좌표별로 따로 둔다 |

이 모델들은 모두 key와 value를 따로 만들고, state를 0에서 시작해 decay시킨다. 우리 layer는 key와 value를 같은 map으로 만들고 state를 단위행렬에서 시작한다. 그래서 제곱해서 읽는 head를 이런 모델에 넣으려면 이 head만 key와 value를 같은 map으로 만들거나, value를 key 공간으로 옮기는 map을 따로 학습해야 한다. key를 길이 1로 맞추는 것과 head 차원이 64에서 128인 것은 우리 layer와 같다.

## 4. 긴 문맥에서 여러 칸을 따라가는 평가

| 평가 | 내용 |
|---|---|
| RULER (Hsieh, 2024, arXiv:2404.06654) | 변수 추적 과제는 체인 1개, 4칸이고 변수 이름 5개를 모두 답하게 한다. RWKV-v5-7B와 Mamba-2.8B는 4K까지 Llama2-7B보다 낮고 8K에서 더 떨어진다. 수치는 그림에만 있다. 칸 수나 체인 수를 늘리면 큰 모델도 크게 떨어진다 |
| Kimi Linear, Qwen3-Next | RULER 평균만 보고한다. Kimi Linear는 128k에서 84.3, Qwen3-Next는 128k에서 96.0, 1M에서 80.3이다. 변수 추적만의 점수는 찾지 못했다 |
| BABILong (Kuratov, 2024, arXiv:2406.10149) | 근거 문장이 둘이나 셋 필요한 질문에서 큰 언어 모델도 85%와 80% 아래이다 |
| Michelangelo (Vodrahalli, 2024, arXiv:2409.12640) | 초록만. linear attention 모델의 점수는 없다 |

## 5. 여기서 고른 순서

1. 이름이 많을 때 학습이 요령에 갇히는 문제를 먼저 푼다. 언어 모델의 문맥에는 개체가 head 차원보다 훨씬 많다. 이 문제가 남아 있으면 뒤의 단계에서 head가 학습되지 않는다. 문헌에서 근거가 있는 처방은 쉬운 문제부터 늘리거나 섞는 학습, 초기화, learning rate 범위, loss를 더 많은 위치에 주는 것이다. 실험 9에서 이들을 한 번에 하나씩 바꿔 비교한다.
2. 그다음은 프로그램 전체에 대한 다음 token 예측으로, 모든 위치에서 읽으며 학습하는 것이다. 문제의 출처 논문과 모든 언어 모델이 이 방식으로 학습한다.
3. 그다음은 key와 value를 따로 만드는 표준 head 옆에서 value를 key 공간으로 옮기는 map을 학습하는 것이다. 3절의 모델들에 넣으려면 필요하고, 발표된 연구를 찾지 못했다.
4. 작은 hybrid 언어 모델은 마지막이다. 3절의 모델들은 RULER 평균이 이미 80에서 96이고 RULER의 변수 추적은 4칸뿐이다. 이익을 보이려면 칸 수와 체인 수가 더 많은 평가를 먼저 정해야 한다.
