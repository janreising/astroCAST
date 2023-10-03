# astroCAST

astroCAST is a Python package designed for analyzing calcium fluorescence events in astrocytes. It provides various functionalities to process and explore calcium imaging data, identify events, and perform spatiotemporal analysis.

## Features

- Fast data preparation
- Efficient event detection algorithm
- Visualization tools for exploring calcium imaging data and events
- Calculation of spatial statistics and cluster analysis
- Modularity detection and analysis of astrocyte modules

## Summary
Astrocytic calcium event analysis is challenging due to its complex nature, spatiotemporally overlapping events, and variable lengths. Our Astrocytic Calcium Signaling Toolkit (astroCAST) addresses these challenges by efficiently implementing event detection and variable length clustering. Leveraging dynamic thresholding, astroCAST captures diverse calcium fluctuations effectively. Designed for modularity, parallelization, and memory-efficiency, it enables fast and scalable event detection on large datasets.

## Installation
You can install astroCAST using pip: 
```shell 
pip install astrocast[all]
```

Alternatively, clone this repository and install it locally:
```shell
pip install poetry
poetry install --with vis
```

MacOS users experience a dependency issue which can be fixed by installing dependencies manually:
```shell
pip install poetry
poetry install
pip install umap-learn napari-plot==0.1.5 pyqt5-tools 
```

For more detailed examples and usage instructions, please refer to to the companion paper [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4491483) (currently in preprint).

## Documentation

[//]: # (The documentation for astroCAST can be found here.)
We are currently working on the documentation. Please consult the `notebooks/` folder for example usage in the meantime.

## Contributing

Contributions to astroCAST are welcome! If you encounter any issues, have suggestions, or would like to contribute new features, please submit a pull request or open an issue in the GitHub repository.

## License

astroCAST is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contact

For any inquiries or questions, please contact [Jan Philipp Reising](mailto:jan.reising@ki.se) or [Ana Cristina González](mailto:ana.cristina.gonzalez.sanchez@ki.se).
