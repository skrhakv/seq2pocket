# Seq2Pocket: From Sequence-Level pLM Predictions to 3D Structural Pockets
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/skrhakv/seq2pocket/blob/master/LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset%20on%20Zenodo-10.5281%2Fzenodo.18271516%20-blue.svg)](https://zenodo.org/records/18271517)

**Seq2Pocket** is a framework for **protein ligand binding site (LBS) prediction** that maps sequence-level predictions to 3D structural pockets. The method utilizes finetuned protein language model (pLM) to identify binding residues and restores pocket continuity through a two-step refinement process:
1. **Embedding-Supported Smoothing:** An additional classifier leverages latent embeddings of neighboring residues to fill in gaps and resolve spatial incompletion inherent in independent residue-wise predictions.

2. **Structure-based Clustering:** A surface-based clustering that utilizes Solvent Accessible Surface (SAS) points to group refined predictions into distinct, biologically relevant 3D pockets.

<p align="center" >
  <img src="https://raw.githubusercontent.com/skrhakv/seq2pocket/refs/heads/master/img/1bygA.png?raw=true" width=1000/>
</p>

***The Seq2Pocket Pipeline.** Our framework takes residue-level probabilities from a finetuned pLM, applies an embedding-supported smoothing classifier to restore pocket continuity, and utilizes a surface-based clustering to define final 3D binding regions. Here, the finetuned pLM correctly identifies the Staurosporine inhibitor binding site on the human C-terminal Src kinase (PDB ID = 1bygA).*

## Availability
You can check out the model for cryptic binding site prediction using on this website: [https://cryptoshow.cz/](https://cryptoshow.cz/), the model for general binding site prediction can be accessed here: [https://seq2pocket.projekty.ms.mff.cuni.cz/](https://seq2pocket.projekty.ms.mff.cuni.cz/).

## Run Locally & Installation
Refer to the `/tutorial` folder for running the model locally.

To run the scripts, it might be useful to install the packages specified in `requirements.txt`.

To generate SAS points, update your biopython's `SASA.py` with the updated version from this repository (`/SASA.py`).


## Framework Components
The **Seq2Pocket** workflow is divided into three main stages:
1. **pLM Fine-tuning:** Scripts for fine-tuning ESM2-3B on ligand binding tasks.
2. **Smoothing Classifier:** A module that uses latent embeddings of a residue and its neighbors to decide if additional residues should be included in a pocket, improving structural completeness.
3. **SAS Clustering**: An optimized clustering approach using Solvent Accessible Surface (SAS) points to group predicted residues into distinct 3D pockets.


## Repository Structure
The source code is located in the `src/` folder, which contains the following subfolders:
1. `data-extraction`: Scripts for processing dataset files.
2. `pLM-training`: Training scripts for finetuning the ESM2-3B models for General Binding Site (GBS) prediction.
3. `smoothing-classifier`: Implementation of the embedding-supported smoothing classifier, including training logic.
4. `evaluation`: Benchmarking scripts that utilize out-of-the-box `scikit-learn` clustering algorithms (e.g., DBSCAN, Mean Shift).
5. `clustering`: Implementation of the proposed clustering methodology: MeanShift applied to Solvent Accessible Surface (SAS) points.
6. `stats`: Calculation of data and figures for the manuscript.
7. `visualizations`: Scripts for visualizing pockets in pyMOL.

The enhancement of the sc-PDB dataset is implemented in separate branch named ['scPDB_enhancement' in the CryptoBench repository](https://github.com/skrhakv/CryptoBench/tree/scPDB_enhancement).


## Data and Materials

- scPDB-Enhanced: [Zenodo repository](https://zenodo.org/records/18271517)
  - An updated version of the scPDB dataset containing experimentally observed binding sites and ions omitted in the original release.
- Data and model weights: [Storage link](https://s3.cl4.du.cesnet.cz/a93fcece52e6da0dd335b4459d47b0aebb74836b:share/seq2pocket.tar.gz)

## Contact us
If you have any questions regarding the usage of the framework, comparing your method against the benchmark, or if you have any suggestions, please feel free to contact us by raising [an issue!](https://github.com/skrhakv/seq2pocket/issues)

## How to cite
If you find our work useful, please cite [the paper](https://www.biorxiv.org/content/10.64898/2026.01.28.702257v1):

- *Vít Škrhák, Lukáš Polák, Marian Novotný, and David Hoksza. 2026. Seq2Pocket: Augmenting protein language models for spatially consistent binding site prediction. bioRxiv. [https://doi.org/10.64898/2026.01.28.702257](https://doi.org/10.64898/2026.01.28.702257)*


or, if you prefer the `BibTeX` format:

```
@article {2026.01.28.702257,
	author = {Škrhák, Vít and Polák, Lukáš and Novotný, Marian and Hoksza, David},
	title = {Seq2Pocket: Augmenting protein language models for spatially consistent binding site prediction},
	elocation-id = {2026.01.28.702257},
	year = {2026},
	doi = {10.64898/2026.01.28.702257},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/01/31/2026.01.28.702257},
	eprint = {https://www.biorxiv.org/content/early/2026/01/31/2026.01.28.702257.full.pdf},
	journal = {bioRxiv}
}
```


## License
This source code is licensed under the [MIT license](https://github.com/skrhakv/seq2pocket/blob/master/LICENSE).
