# L2BR: Learning to Block Relocation

This repository provides the implementation, trained models, and benchmark datasets used in the following paper:

**A Unified Learning Framework for the Block Relocation Problem: A New Performance Baseline**  
Woo-Jin Shin, Ji-Kwang Jung, Sang-Hyun Cho, Inguk Choi, Shunji Tanaka, and Hyun-Jung Kim  
*European Journal of Operational Research*, 2026.  
https://doi.org/10.1016/j.ejor.2026.09.016

## Overview

L2BR is a unified learning framework for both the restricted and unrestricted Block Relocation Problem (BRP).  
The framework sequentially combines imitation learning (IL) and self-improvement learning (SIL) and uses a size-agnostic neural architecture for solving BRP instances of different scales.

## Repository Structure

- `IL/`: files and trained models related to imitation learning
- `RL/`: files and trained models related to reinforcement learning
- `SIL/`: files and trained models related to self-improvement learning
- `datasets/`: benchmark instances used in the computational experiments
- `train_IL.py`: training script for imitation learning
- `train_RL.py`: training script for reinforcement learning
- `train_SIL.py`: training script for self-improvement learning
- `test.py`: evaluation script for benchmark instances

## Benchmark Datasets

The `datasets/` directory contains three benchmark sets used in our computational experiments:

- **CVS**  
  Caserta, M., Schwarze, S., and Voß, S. (2012).  
  *A mathematical formulation and complexity considerations for the blocks relocation problem.*  
  European Journal of Operational Research, 219(1), 96-104.

- **ZQLZ**  
  Zhu, W., Qin, H., Lim, A., and Zhang, H. (2012).  
  *Iterative deepening A* algorithms for the container relocation problem.*  
  IEEE Transactions on Automation Science and Engineering, 9(4), 710-722.

- **LL**  
  Lee, Y., and Lee, Y.-J. (2010).  
  *A heuristic for retrieving containers from a yard.*  
  Computers & Operations Research, 37(6), 1139-1147.

These benchmark instances were originally introduced and publicly distributed by the respective authors. They are included here to facilitate reproducibility of the computational experiments reported in our paper.

We do not claim ownership of these datasets. All rights and credits remain with the respective original authors. The benchmark datasets are not covered by the MIT License that applies to the source code in this repository.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{shin2026l2br,
  title={A unified learning framework for the block relocation problem: A new performance baseline},
  author={Shin, Woo-Jin and Jung, Ji-Kwang and Cho, Sang-Hyun and Choi, Inguk and Tanaka, Shunji and Kim, Hyun-Jung},
  journal={European Journal of Operational Research},
  year={2026},
  doi={10.1016/j.ejor.2026.09.016}
}
```

## License

The source code in this repository is released under the MIT License.

The benchmark datasets in `datasets/` are third-party materials and are not covered by the MIT License of this repository.
