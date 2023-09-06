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
[//]: # (You can install astroCAST using pip: ```shell pip install astroCAST ```)

You can install astroCAST by cloning the github repository and installing it locally. We will making it available through pip soon.
```shell
pip install -e .
```

Please note that astroCAST has several dependencies. You can refer to the `requirements.txt` file for a complete list of dependencies.

## Usage

Here's a minimal example of how to use astroCAST:

```python
from astrocast import preparation, detection, analysis, helper, clustering

# Prepare the data
converter = preparation.Input()
converter.run(input_path='path/to/data', output_path='path/to/output.h5')

# Detrend the data
delta = preparation.Delta('path/to/output.h5')
dF = delta.run()
delta.save('path/to/output.h5')

# Detect events
detector = detection.Detection('path/to/output.h5')
data = detector.run()
events = analysis.Events('path/to/output.roi')

# Data exploration (examplary)
events.get_summary_statistic()

# Event preparation
filter_instructions = {"event_column": (1, 20)}
events.filter(filter_instructions, inplace=True)

normalization_instructions = {
    0: ["subtract": {"mode": "mean"}], 1: ["divide": {"mode": "std"}]
}
events.normalize(normalization_instructions, inplace=True)

# Unify length
events.enforce_length(min_length=10, max_length=5, inplace=True)

# Clustering
distance = clustering.Distance()
corr = distance.get_correlation(events, correlation_type="dtw")

linkage = clustering.Linkage()
barycenters, cluster_lookup = linkage.get_barycenters(events, z_threshold=2, distance_matrix=corr)
events.add_clustering(cluster_lookup, column_name="bary_cluster")
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
