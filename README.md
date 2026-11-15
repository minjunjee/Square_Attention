# Square_Attention: 상태를 제곱해서 읽는 linear attention

2026년 가을학기 대형언어모델개론 팀 프로젝트. DeltaNet 같은 linear attention은 layer마다 상태 S를 한 번만 읽어서 체인을 k칸 따라가려면 layer가 k개 필요하다. 읽기 전에 S를 r번 제곱하면 layer 하나가 2^r칸을 따라간다는 주장을 서버에서 검증하고 있다.

## 먼저 읽을 것

| 파일 | 내용 |
|---|---|
| `draft/DRAFT.md` | 초안. 지금까지의 결과와 해석이 가장 최신으로 정리되어 있다 |
| `SERVER_POC_PLAN.md` | 실험 계획과 실험 전에 정한 판정 기준 |
| `SERVER_RESULTS.md` | 서버 실행 기록. 환경, 사고, 실험 0과 1, 4의 상세 |
| `PAPER_CHECK.md` | 문제의 출처 논문(Wu, Geiger, Millière, ICML 2025)과의 대조 |
| `results/tables_*.md` | 실험별 seed 단위 표. `summarize.py`로 다시 만든다 |
| `results/figures/` | 그림. `make_server.py`로 다시 만든다 |

## 코드

| 파일 | 역할 |
|---|---|
| `poc_fair.py` | 문제 생성, 제안 layer, DeltaNet, Transformer, LSTM, 두 시험. 실험의 모든 학습이 이 파일 하나로 돈다 |
| `cost_read.py`, `squared_read_triton.py`, `fused_read.py` | GPU 비용 측정과 Triton 커널. 커널 쪽은 `.venv_tc` 같은 Triton이 동작하는 환경이 필요하다 |
| `precision_eval.py` | TF32와 bf16에서 정확도가 유지되는지 |
| `summarize.py`, `judge_e2.py`, `choose_lr.py`, `choose_fix.py` | 표, 실험 2 판정, learning rate와 실험 4 설정 고르기 |
| `run_gpu_pool.sh`, `run_gpu_pool_claim.sh`, `run_after_*.sh` | 작업 목록 `jobs/*.txt`를 GPU에 나눠 돌리고 실험을 이어 붙인다 |
| `tools/` | 상태 확인, 대기, 진단 스크립트 |

## 돌리는 법

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch numpy matplotlib --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
cp gpu_allowlist.example.txt gpu_allowlist.txt    # 이 기계에서 써도 되는 GPU의 UUID만 남긴다
CUDA_VISIBLE_DEVICES=<GPU UUID> .venv/bin/python poc_fair.py --model tied --read square --tap_attn 8 \
  --tap_sharp 10 --gate_ctx 1 --r_hi 5 --r_eval 6 --steps 30000 --lr 1e-3 --seed 0
```

**GPU 지정 주의.** 이 서버는 GPU 0에서 3이 MIG로 나뉘어 있어서, `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 써도 CUDA 번호와 nvidia-smi 번호가 다르다. CUDA 번호 4에서 7은 물리 GPU 0과 1의 MIG 인스턴스이다. GPU는 반드시 UUID로 지정한다. `gpu_allowlist.txt`가 있으면 `gpu_guard.py`가 목록 밖의 지정을 CUDA 초기화 전에 거부한다.

## 상태

이 branch는 11월 16일 01:30 KST 기준의 스냅샷이다. 실험 2(충분히 학습한 표준 모델과의 비교)와 변수 추적 변형 실험이 서버에서 계속 돌고 있다. 체크포인트(`ckpt/`)와 실행별 로그는 용량 때문에 올리지 않았고, 모든 실행의 숫자는 `results/res_server.jsonl`에 있다. 일부 스크립트에는 서버의 절대 경로 `/home/minjun_dev/poc_squared_read`가 남아 있다.
