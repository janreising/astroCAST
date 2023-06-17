import argparse
import json
from pathlib import Path

from astroCAST.analysis import Events
from astroCAST.clustering import DTW_Linkage

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-lc", "--cachepath", required=False, default=None, help="Local cache directory")
    parser.add_argument("-ed", "--eventdir", required=True, default=None, help="Path to event directory")
    parser.add_argument("--zthr", type=int, default=2, help="Z threshold")

    parser.add_argument("--correlation", type=str, default="pearson", help="Correlation type: (pearson, dtw)")

    parser.add_argument("--distanceparam", type=str, nargs="+", default={}, help="Distance Matrix Parameters, space separated. e.g. 'penalty,None' 'psi,None'")
    parser.add_argument("--linkageparam", type=str, nargs="+", default={}, help="Linkage Matrix Parameters, space separated. e.g. 'penalty,None' 'psi,None'")
    parser.add_argument("--clusteringparam", type=str, nargs="+", default={}, help="Clustering Parameters, space separated. e.g. 'penalty,None' 'psi,None'")
    parser.add_argument("--baryparam", type=str, nargs="+", default={}, help="Barycenter Calculation Parameters, space separated. e.g. 'penalty,None' 'psi,None'")

    args = parser.parse_args()

    # load events
    event_dir = Path(args.eventdir)
    events = Events(event_dir=event_dir)

    # run linkage analysis
    dtw = DTW_Linkage(cache_path=args.cachepath)
    barycenters, cluster_lookup_table = dtw.get_barycenters(events, z_threshold=args.zthr,
                                                            distance_type=args.correlation,
                                                            param_distance=args.distanceparam,
                                                            param_linkage_matrix=args.linkageparam,
                                                            param_clustering=args.clusteringparam,
                                                            param_barycenter=args.baryparam,
                                                            )

    # save results
    barycenters.to_csv(event_dir.joinpath("script_barycenters.csv"))
    with open(event_dir.joinpath("script_cluster_lookup_table.csv").as_posix(), 'w') as fp:
        json.dump(cluster_lookup_table, fp)
