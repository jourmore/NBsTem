# Thermostability Prediction Powered by Synergistic Deep Learning at Experimental and Theoretical Levels for Nanobodies
## [_Webserver Link_](https://www.nbscal.online/)
<p align="center">
    <img align="center" src="https://github.com/jourmore/NBsTem/blob/master/GA.png" width="500" alt="logo"/>
</p>

- This is the official repository of NBsTem_Tm & NBsTem_Q, two deep learning models designed for thermostability prediction of nanobodies (VHH).
- You can also access [NBsTem Webserver](https://www.nbscal.online/) for thermostability prediction online.

## 1.Setup

Clone this repository and install the package locally:
```bash
$ git clone git@github.com:jourmore/NBsTem.git
$ cd NBsTem_local
$ pip install -r requirements.txt
```

## 2.Usage

```bash
python app_uncertainty.py -i in.fasta
python app_uncertainty.py -t QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS
```

```bash
*usage: python app_uncertainty.py [-h] [-i I] [-o O] [-t T] [-seed SEED] [-device DEVICE]

optional arguments:
  -h, --help      show this help message and exit
  -i I            Input path with fasta format. [Such as: ./in.fasta]
  -o O            Output file name when input is fasta format. [Default: "Output-NBsTem-[Year]-[Month]-[Day].csv"
  -t T            Input one sequecne with text format. [Default:
                  QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS]
  -seed SEED      Random seed for torch, numpy, os. [Default: 42]
  -device DEVICE  Device: cpu, cuda. [Default: auto]
```

### Example

- Example (Using default parameters and example sequences):

```bash
python app_uncertainty.py -i example.fasta -o output.csv
```

- Terminal output message:

```bash
******************************************************************
**                                                              **
**  NBsTem v.2025 Thermostability prediction for Nanobody/VHH.  **
**                                                              **
**                 https://www.nbscal.online/                   **
**                   maojun@stu.scu.edu.cn                      **
******************************************************************

== 1.Use seed: 42
== 2.Device: cuda
== 3.Loading antibody language model: AntiBERTy
== 5.Begin to predict: Tm, Qclass, Specie and Chain
** Calculating Specie and Chain [Fast]
** Calculating Tm:: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [00:04<00:00, 18.33it/s]
** Calculating Qclass:: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 83/83 [00:02<00:00, 29.13it/s]
== 6.Finish ! The results are shown below or you can check file [Tm83test.csv]

                    ID     Tm  Tm_Uncertainty  Qclass Q_Uncertainty Specie                                           Sequence
1                 4W70  73.28            1.50       3     -1.00e-08  Camel  EVQLVESGGGLVQAGDSLRLSATASGRTFSRAVMGWFRQAPGKERE...
2                 5SV3  69.77            2.55       3      7.22e-01  Camel  EVQLVESGGGLVQAGDSLRLSCTASGRTLGDYGVAWFRQAPGKERE...
3                  Nb4  63.79            2.41       3      9.71e-01  Camel  QVQLVESGGGSVQAGGSLRLSCAASGLDIHSYCMTWFRQAPGKERE...
4                  Nb5  68.08            1.94       2     -1.00e-08  Camel  QVQLVESGGGSVQAGGSLRLSCAASGSAISNLYMAWFRQAPGKERE...
5                  Nb6  80.32            2.40       2     -1.00e-08  Camel  HVQLVESGGGSVQAGGSLRLSCEISLYIYSSYCMGWFRQAPGKERE...
..                 ...    ...             ...     ...           ...    ...                                                ...
79  NB-AGT-2-L22A-I72V  67.87            1.84       2      7.22e-01  Camel  QVQLVESGGGLVQAGGSLRASCAASGRTFSSYAMGWFRQAPGKERE...
80  NB-AGT-2-L22A-I72A  69.10            1.55       2      7.22e-01  Camel  QVQLVESGGGLVQAGGSLRASCAASGRTFSSYAMGWFRQAPGKERE...
81            NB-extra  74.07            2.09       3      9.71e-01  Human  EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIGWVRRAPGKGEE...
82      NB-extra-CA-CV  71.00            2.33       3      9.71e-01  Human  EVQLVESGGGLVQPGGSLRLSAAASGFNIKDTYIGWVRRAPGKGEE...
83      NB-extra-CA-CA  71.02            2.31       3      9.71e-01  Human  EVQLVESGGGLVQPGGSLRLSAAASGFNIKDTYIGWVRRAPGKGEE...

[83 rows x 7 columns]

```

## 3.About models

- **NBsTem_Tm**: A model for predicting the melting temperature (Tm) from experiments (nanoDSF, DSF, DSC and CD, etc.).

- **NBsTem_Q**: A model for predicting a new theoretical indicator (Qclass) proposed by us, which is derived from molecular dynamics simulation.

## Citing this work

```bibtex
@article{...,
    Title = {Thermostability Prediction Powered by Synergistic Deep Learning at Experimental and Theoretical Levels for Nanobodies},
    Authors = {Jourmore, Yuanpeng Song, Ming Kong, Yanzhi Guo, Yijing Liu, and Xuemei Pu},
    Journal = {ACS Applied Materials & Interfaces},
    Year= {2025}
}
```
