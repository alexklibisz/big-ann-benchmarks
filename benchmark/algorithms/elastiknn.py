from multiprocessing import cpu_count
from psutil import virtual_memory
from benchmark.algorithms.httpann import HttpANN, HttpANNSubprocess
from benchmark.datasets import DATASETS


class ElastiknnDenseFloatL2Lsh(HttpANN, HttpANNSubprocess):

    def __init__(self, dimension: int, count: int, lsh_L: int, lsh_k: int, lsh_w: int):
        memgb = int(virtual_memory().total / 1e9 / 2)
        cmd = " ".join([
            "java -cp /home/app/ann-benchmarks.jar",
            f"-Xms{memgb}G -Xmx{memgb}G",
            "-XX:+UseG1GC",
            "-XX:G1ReservePercent=25",
            "-XX:InitiatingHeapOccupancyPercent=30",
            "-XX:+HeapDumpOnOutOfMemoryError",
            "-Dcom.sun.management.jmxremote.ssl=false",
            "-Dcom.sun.management.jmxremote.authenticate=false",
            "-Dcom.sun.management.jmxremote.local.only=false",
            "-Dcom.sun.management.jmxremote.port=9091",
            "-Dcom.sun.management.jmxremote.rmi.port=9091",
            "-Djava.rmi.server.hostname=localhost",
            "com.elastiknn.annb.Server",
            "--datasets-path /home/app/data",
            "--index-path /tmp",
            "--algorithm l2lsh",
            "--vector-type float32",
            f"--index-args '[{dimension}, {lsh_L}, {lsh_k}, {lsh_w}]'",
            f"--count {count}",
            "--port 8080",
            f"--parallelism {cpu_count()}"
        ])
        self.base_name = f"elastiknn-dense-float-l2-lsh-{lsh_L}-{lsh_k}-{lsh_w}"
        HttpANNSubprocess.__init__(self, cmd)
        HttpANN.__init__(self, server_url="http://0.0.0.0:8080", start_seconds=3, name=self.base_name)

    def fit(self, dataset):
        # ds = DATASETS[dataset]()
        # for v in ds.get_dataset()[:10]:
        #     print(v[:10])
        super().fit(dataset)

    def set_query_arguments(self, *query_args):
        candidates, probes = query_args
        self.name = f"{self.base_name}_{candidates}_{probes}"
        super().set_query_arguments(candidates, probes)
