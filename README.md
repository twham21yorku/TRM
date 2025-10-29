# DeepGlobe-TRM: Step-wise Adaptive Refinement for Land Cover Segmentation

This repository is a **production-ready scaffold** to train and evaluate a **TRM-style iterative refiner with learned halting** on the **DeepGlobe Land Cover** dataset.

> **Dataset facts (DeepGlobe 2018):**
> * RGB satellite images, **2448×2448**, train/val/test ≈ **803/171/172**.  
> * **7 label colors**: Urban (0,255,255), Agriculture (255,255,0), Rangeland (255,0,255), Forest (0,255,0), Water (0,0,255), Barren (255,255,255), Unknown (0,0,0). Unknown is **ignored** in evaluation.  
> * File names follow pattern: `<id>_sat.jpg` (image), `<id>_mask.png` (mask).  
> * Due to compression, labels are recommended to be **binarized per channel at 128** before mapping to classes.

Citations: DeepGlobe paper (CVPR’18 workshops) and a widely used DeepGlobe repo that documents the class colors & ignore rule.

---

## Why this project?
- **Small shared refiner** run **multiple steps** (e.g., 2–6) to improve predictions, instead of using many fixed-depth heads.
- **Learned halting (ACT)** decides **when to stop**; easy tiles use fewer steps, hard tiles get more.
- Strong **tracing/logging** and neat **file structure**, with patch index, color-mapping, and evaluation utilities ready.

---

## What's New In This Scaffold
- Local dataset/config wiring: `Dataset/DeepGlobe`와 `project_root` 경로를 반영.
- 패치 인덱서 확장: `train/val` 뿐 아니라 `train/valid`도 지원, 검증 마스크가 없으면 `train`에서 자동 샘플링.
- 학습 개선: 워밍업+코사인 스케줄, CSV 로그 확장(`lr`, `val_avg_steps`, `val_bf1`), 메타데이터 JSON 기록.
- 추론 개선: 오버랩 타일링 지원(`--stride`), ACT로 조기 종료 적용.
- 유틸 추가:
  - FLOPs/Latency 측정(THOP 기반 근사 + CUDA 타이머).
  - ECE/Temperature 보정(픽셀 단위).
  - Conformal‑halting 스타일 τ 캘리브레이션과 q 통계 저장.
- 반복 실험 스크립트: 5개 시드 자동 실행(`scripts/run_sweep.sh`).

---

## File tree (key files)

```
deepglobe_trm/
  configs/
    deepglobe_default.yaml
  src/
    data/
      deepglobe_dataset.py      # RGB→class mapping, patch dataset, transforms
      transforms.py             # dihedral, color jitter, noise
      patchify.py               # create patch index CSVs
      label_codec.py            # robust color→class mapping utilities
    models/
      blocks.py                 # depthwise-separable conv blocks
      trm_refiner.py            # encoder + iterative refiner + halting heads
    losses.py                   # CE (deeply supervised) + halting + ponder
    metrics.py                  # mIoU, class IoU, boundary F1
    utils.py                    # config, seeding, logging, checkpointing
    train.py                    # full training loop with mixed precision
    infer.py                    # whole-image tiling inference, ACT enabled
    calibrate_halt.py           # τ 스윕 + conformal-style 추천 τ + q 통계
    tools/
      efficiency.py             # FLOPs(근사, THOP) + latency 측정
      calibration.py            # 픽셀 단위 ECE + temperature scaling
  scripts/
    prepare_deepglobe.py        # build patch index CSVs from raw dataset
    run_sweep.sh                # 5개 시드 자동 학습 실행
  experiments/                  # outputs
  requirements.txt
  README.md
```

---

## Quickstart

1) **Install** (Python ≥3.9)
```bash
pip install -r requirements.txt
```

2) **Prepare DeepGlobe**  
If you already have the dataset at `Dataset/DeepGlobe` (with `train/valid/test`), you're set. Otherwise arrange like:
```
/data/DeepGlobe/
  train/    # or train+val (or train+valid)
    000000_sat.jpg
    000000_mask.png
    ...
  val/ or valid/  # optional; otherwise script will split from train
```

3) **Index patches**
```bash
python -m src.data.patchify --cfg configs/deepglobe_default.yaml --root /home/twham21/Workspace/Ph.D/Dataset/DeepGlobe
```
This writes CSVs into `data/index/` (train/val). The script honors `dataset.train_split` and `dataset.val_split` in the config (supports `valid`). You can change patch size/stride in the config.
Note: In many DeepGlobe mirrors, `valid/` and `test/` lack masks; in that case the script will automatically sample a validation list from `train/`.

4) **Train**
```bash
python -m src.train --cfg configs/deepglobe_default.yaml
```

5) **Calibrate halting (optional)**
```bash
python -m src.calibrate_halt --cfg configs/deepglobe_default.yaml
```

6) **Infer on full images**
```bash
python -m src.infer --cfg configs/deepglobe_default.yaml --input /home/twham21/Workspace/Ph.D/Dataset/DeepGlobe/valid --out out_masks/
```

---

## One-Command CLI (scripts/trm.py)
반복되는 명령을 하나로 묶은 통합 실행기를 추가했습니다.

예시:
- 패치 인덱스 생성: `python scripts/trm.py prepare --cfg configs/deepglobe_default.yaml`
- 학습: `python scripts/trm.py train --cfg configs/deepglobe_default.yaml --seed 42`
- 추론: `python scripts/trm.py infer --cfg configs/deepglobe_default.yaml --input Dataset/DeepGlobe/valid --out out_masks --stride 128`
- Halting 캘리브레이션: `python scripts/trm.py halt-cal --cfg configs/deepglobe_default.yaml --checkpoint experiments/run1/epoch_80.pt --delta 0.1`
- ECE/온도 보정: `python scripts/trm.py ece --cfg configs/deepglobe_default.yaml --checkpoint experiments/run1/epoch_80.pt`
- FLOPs/Latency: `python scripts/trm.py efficiency --cfg configs/deepglobe_default.yaml --steps 6`
- 5-시드 스윕: `python scripts/trm.py sweep --cfg configs/deepglobe_default.yaml`

---

## Dataset & Preprocessing
- 라벨 색상→클래스 매핑(압축 노이즈 강건): Urban (0,255,255), Agriculture (255,255,0), Rangeland (255,0,255), Forest (0,255,0), Water (0,0,255), Barren (255,255,255), Unknown (0,0,0→ignore=255).
- 패치: 기본 `256×256`, stride=256(겹침 없음). 겹침을 원하면 `dataset.patch.stride`를 작게 설정(예: 128).
- Unknown 비율 90% 초과 패치는 제거(`filter_unknown_ratio=0.9`).
- 증강: dihedral(8가지 회전/좌우반전) + 약한 밝기/대비/감마 지터.

## Patch Indexing Details
- 명령: `python -m src.data.patchify --cfg <yaml> --root <dataset_root>`
- 분할 폴더 이름은 설정의 `dataset.train_split`/`dataset.val_split`을 따릅니다(train/val 또는 train/valid).
- `valid/`/`test/`에 마스크가 없으면, `train`에서 15%를 자동 샘플링해 검증 리스트를 생성합니다.
- 재실행 시, 기존 인덱스가 설정과 일치하면 자동으로 건너뜁니다(빠른 재시작). 강제 재생성은 `--force`를 사용하세요.

## Training
- 명령: `python -m src.train --cfg configs/deepglobe_default.yaml [--seed 42] [--out_dir experiments/run1]`
- 스케줄: 워밍업(`scheduler.warmup_epochs`) 후 코사인(에폭 기준). AMP, grad clip, AdamW 사용.
- 손실: 최종 CE + 중간 CE(깊은 감독, alpha) + halting BCE(beta) + ponder(옵션, gamma).
- 검증: mIoU(Unknown 제외), 경계 F1(BF score), 평균 사용 스텝(`val_avg_steps`).
- 로그: `train_log.csv` 컬럼 → `epoch, lr, train_loss, val_loss, val_miou, val_avg_steps, val_bf1`.
- 메타: `metadata.json`에 환경/커밋/cfg 저장.

## Inference (Tiling & ACT)
- 명령: `python -m src.infer --cfg <yaml> --input <folder> --out <out_dir> [--stride 128]`
- 패치 타일링 후 스티칭. q-head 기반 ACT로 `tau`를 만족하는 가장 빠른 스텝을 사용.
- 겹침 타일링은 `--stride < patch`로 활성화(예: `--stride 128`).

## Efficiency (FLOPs & Latency)
- 명령: `python -m src.tools.efficiency --cfg configs/deepglobe_default.yaml --steps 6`
- 출력: 파라미터(M), Approx FLOPs(G MACs; THOP), Latency/patch(ms; warmup 후 평균±표준편차).
- 주의: THOP은 일부 연산/바이어스 미포함, 커스텀 모듈 0 처리 가능 → 절대치보다 상대 비교 지향.
- TRM 총 연산량 근사: `per-step FLOPs × 평균 사용 스텝(val_avg_steps)`.

## Calibration (ECE & Temperature)
- 명령: `python -m src.tools.calibration --cfg <yaml> --checkpoint <ckpt.pt>`
- 기능: 픽셀 단위 ECE 계산, 검증셋으로 온도 `T` 피팅, ECE before/after 및 `T` 저장(JSON).
- 필요 시 추론 파이프라인에서 `logits/T` 적용.

## Halting Calibration (Conformal-style)
- 명령: `python -m src.calibrate_halt --cfg <yaml> [--delta 0.1] [--save_csv q_stats.csv]`
- 내용: τ 스윕 결과(mIoU/avg_steps), q 분포로 Coverage@δ 만족 τ 추천(근사: `quantile(q, 1-δ)`), 스텝별 q 통계 CSV 옵션.

## Multi-seed Runs
- `bash scripts/run_sweep.sh configs/deepglobe_default.yaml`로 5회 반복 실행. 각 결과는 `experiments/run_seed_*/train_log.csv`에 저장.

## Config Guide
- `dataset.root/patch.size/stride/filter_unknown_ratio/train_split/val_split`
- `training`(batch, workers, max_epochs, amp, grad_clip_norm, save_every)
- `optimizer`(lr, weight_decay, betas), `scheduler.warmup_epochs`
- `model`(in_channels, num_classes=6, width, steps[t_min,t_max,tau,act], pixel_act)
- `logging.out_dir`, `project_root`

## Notes
- Unknown (0,0,0) pixels are mapped to `ignore_index=255` and excluded from loss/metrics.
- Color→class mapping is **robust to compression artifacts** via channel-wise binarization at 128 (per DeepGlobe guidance).
- Set `model.steps.act=false` to disable learned halting and run fixed steps during development.

---

## License
MIT
