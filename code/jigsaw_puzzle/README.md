## Environments

- Window 10
- NVIDIA GeForce RTX 3060
- Conda 4.12.0
- Python 3.8.13
- Pytorch 1.11.0

## Get Started

- Each script except `train.ipynb` is baseline code
- In `train.ipynb`, Change the some directory(ex. data path, permut set path) and just run all cells
    - configuration is at the bottom cell
- If you don’t have permutation set, You must run `permutations.ipynb` to create a permutation set.
    - I recommend that If you want to only test with under 200 images, run with `N=3`
        - `N` is number of permutation set
