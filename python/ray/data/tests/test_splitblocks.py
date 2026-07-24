import numpy as np
import pyarrow as pa
import pytest

import ray
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import (
    AdditionalSplitBudget,
    _apply_additional_split_budget,
    _split_blocks,
    _splitrange,
)
from ray.data.block import BlockAccessor, BlockMetadata
from ray.data.datasource import Datasource
from ray.data.datasource.datasource import ReadTask
from ray.data.tests.conftest import *  # noqa
from ray.data.tests.conftest import (
    CoreExecutionMetrics,
    assert_core_execution_metrics_equals,
    get_initial_core_execution_metrics_snapshot,
)
from ray.tests.conftest import *  # noqa


def test_splitrange():
    def f(n, k):
        assert _splitrange(n, k) == [len(a) for a in np.array_split(range(n), k)]

    f(0, 1)
    f(5, 1)
    f(5, 3)
    f(5, 5)
    f(5, 10)
    f(50, 1)
    f(50, 2)
    f(50, 3)
    f(50, 4)
    f(50, 5)


def test_split_blocks():
    def f(n, k):
        table = pa.Table.from_arrays([np.arange(n)], names=["value"])
        in_blocks = [table]
        out_blocks = list(_split_blocks(in_blocks, k))
        sizes = [BlockAccessor.for_block(b).num_rows() for b in out_blocks]
        expected = [len(a) for a in np.array_split(range(n), min(k, n))]
        assert sizes == expected

    f(5, 1)
    f(5, 3)
    f(5, 5)
    f(5, 10)
    f(50, 1)
    f(50, 2)
    f(50, 3)
    f(50, 4)
    f(50, 5)


def _block_sizes(blocks):
    return [BlockAccessor.for_block(block).num_rows() for block in blocks]


def _task_ctx(task_idx):
    return TaskContext(task_idx=task_idx, op_name="Read")


def _additional_split_budget(
    additional_split_budget,
    num_tasks=1,
):
    return AdditionalSplitBudget(
        additional_split_budget,
        num_tasks,
    )


class _UnderestimatedMultiBlockDatasource(Datasource):
    """Two read tasks whose runtime blocks are larger than their metadata estimate."""

    def __init__(self, blocks_per_task):
        self._blocks_per_task = blocks_per_task

    def estimate_inmemory_data_size(self):
        # Deliberately underestimate runtime bytes so the optimizer installs an
        # additional split budget.
        return 2

    def get_read_tasks(
        self,
        parallelism,
        per_task_row_limit=None,
        data_context=None,
    ):
        tasks = []
        blocks_per_task = self._blocks_per_task
        for task_idx in range(2):

            def read_fn(task_idx=task_idx, blocks_per_task=blocks_per_task):
                for block_idx in range(blocks_per_task):
                    start = 10 * (task_idx * blocks_per_task + block_idx)
                    yield pa.table({"value": range(start, start + 10)})

            tasks.append(
                ReadTask(
                    read_fn,
                    BlockMetadata(
                        num_rows=blocks_per_task * 10,
                        # Deliberately underestimate runtime bytes so the
                        # optimizer installs an additional split budget.
                        size_bytes=1,
                        input_files=None,
                        exec_stats=None,
                    ),
                    per_task_row_limit=per_task_row_limit,
                )
            )
        return tasks


@pytest.mark.parametrize(
    "blocks_per_task,override_num_blocks,expected_block_sizes",
    [
        (1, 6, [3, 3, 3, 3, 4, 4]),
        (3, 4, [5, 5, 5, 5, 10, 10, 10, 10]),
    ],
    ids=[
        "one-block-reads-hit-target",
        "natural-fanout-adds-only-budget",
    ],
)
def test_additional_split_budget_e2e(
    ray_start_10_cpus_shared,
    restore_data_context,
    blocks_per_task,
    override_num_blocks,
    expected_block_sizes,
):
    # Each Arrow block contains 80 data bytes. A 64-byte target shapes every
    # yielded datasource block into a separate natural runtime block.
    ray.data.DataContext.get_current().target_max_block_size = 64

    ds = ray.data.read_datasource(
        _UnderestimatedMultiBlockDatasource(blocks_per_task),
        override_num_blocks=override_num_blocks,
    ).materialize()
    blocks = ray.get(ds.get_internal_block_refs())

    assert sorted(_block_sizes(blocks)) == expected_block_sizes
    assert ds.count() == blocks_per_task * 2 * 10


def test_additional_split_budget_one_block_exact():
    output = list(
        _apply_additional_split_budget(
            [pa.table({"value": range(10)})],
            _task_ctx(0),
            _additional_split_budget(
                additional_split_budget=4,
            ),
        )
    )

    assert _block_sizes(output) == [2, 2, 2, 2, 2]


def test_additional_split_budget_is_bounded():
    blocks = [pa.table({"value": range(i * 10, (i + 1) * 10)}) for i in range(3)]
    output = list(
        _apply_additional_split_budget(
            blocks,
            _task_ctx(0),
            _additional_split_budget(
                additional_split_budget=2,
            ),
        )
    )

    assert _block_sizes(output) == [4, 3, 3, 10, 10]
    assert output[-2] is blocks[-2]
    assert output[-1] is blocks[-1]


def test_additional_split_budget_tiny_blocks():
    output = list(
        _apply_additional_split_budget(
            [
                pa.table({"value": [0, 1]}),
                pa.table({"value": [2]}),
            ],
            _task_ctx(0),
            _additional_split_budget(
                additional_split_budget=5,
            ),
        )
    )

    assert _block_sizes(output) == [1, 1, 1]


def test_additional_split_budget_drops_natural_empty_blocks():
    output = list(
        _apply_additional_split_budget(
            [
                pa.table({"value": []}),
                pa.table({"value": range(4)}),
                pa.table({"value": []}),
            ],
            _task_ctx(0),
            _additional_split_budget(
                additional_split_budget=2,
            ),
        )
    )

    assert _block_sizes(output) == [2, 1, 1]


def test_additional_split_budget_distributes_across_tasks():
    budget = _additional_split_budget(
        additional_split_budget=5,
        num_tasks=2,
    )
    outputs = [
        list(
            _apply_additional_split_budget(
                [pa.table({"value": range(task_idx * 8, (task_idx + 1) * 8)})],
                _task_ctx(task_idx),
                budget,
            )
        )
        for task_idx in range(2)
    ]

    assert [_block_sizes(output) for output in outputs] == [
        [2, 2, 2, 2],
        [3, 3, 2],
    ]


def test_additional_split_budget_ignores_unplanned_task():
    block = pa.table({"value": range(8)})

    output = list(
        _apply_additional_split_budget(
            [block],
            _task_ctx(2),
            _additional_split_budget(
                additional_split_budget=5,
                num_tasks=2,
            ),
        )
    )

    assert len(output) == 1
    assert output[0] is block


def test_additional_split_budget_does_not_look_ahead():
    blocks = [
        pa.table({"value": [0, 1, 2, 3]}),
        pa.table({"value": [4]}),
    ]
    num_consumed = 0

    def iter_blocks():
        nonlocal num_consumed
        for block in blocks:
            num_consumed += 1
            yield block

    output = iter(
        _apply_additional_split_budget(
            iter_blocks(),
            _task_ctx(0),
            _additional_split_budget(
                additional_split_budget=1,
            ),
        )
    )

    # The current natural block is yielded before requesting its successor.
    assert _block_sizes([next(output)]) == [2]
    assert num_consumed == 1
    assert _block_sizes([next(output)]) == [2]
    assert num_consumed == 1
    assert next(output) is blocks[1]


def test_small_file_split(ray_start_10_cpus_shared, restore_data_context):
    last_snapshot = get_initial_core_execution_metrics_snapshot()

    ds = ray.data.read_csv("example://iris.csv", override_num_blocks=1)
    materialized_ds = ds.materialize()
    assert materialized_ds._logical_plan.initial_num_blocks() == 1
    last_snapshot = assert_core_execution_metrics_equals(
        CoreExecutionMetrics(
            task_count={
                "ReadCSV": 1,
            },
        ),
        last_snapshot,
    )

    materialized_ds = ds.map_batches(lambda x: x).materialize()
    assert materialized_ds._logical_plan.initial_num_blocks() == 1
    last_snapshot = assert_core_execution_metrics_equals(
        CoreExecutionMetrics(
            task_count={
                "ReadCSV->MapBatches(<lambda>)": 1,
            },
        ),
        last_snapshot,
    )

    stats = materialized_ds.stats()
    assert "Operator 1 ReadCSV->MapBatches" in stats, stats

    ds = ray.data.read_csv("example://iris.csv", override_num_blocks=10)
    assert ds._logical_plan.initial_num_blocks() == 1
    assert (
        ds.map_batches(lambda x: x).materialize()._logical_plan.initial_num_blocks()
        == 10
    )
    last_snapshot = assert_core_execution_metrics_equals(
        CoreExecutionMetrics(
            task_count={
                "MapBatches(<lambda>)": 10,
                "ReadCSV->SplitBlocks(additional=9)": 1,
            },
        ),
        last_snapshot,
    )

    assert ds.materialize()._logical_plan.initial_num_blocks() == 10
    last_snapshot = assert_core_execution_metrics_equals(
        CoreExecutionMetrics(
            task_count={
                "ReadCSV->SplitBlocks(additional=9)": 1,
            },
        ),
        last_snapshot,
    )

    ds = ray.data.read_csv("example://iris.csv", override_num_blocks=100)
    assert ds._logical_plan.initial_num_blocks() == 1
    assert (
        ds.map_batches(lambda x: x).materialize()._logical_plan.initial_num_blocks()
        == 100
    )
    assert ds.materialize()._logical_plan.initial_num_blocks() == 100

    ds = ds.map_batches(lambda x: x).materialize()
    stats = ds.stats()
    assert "Operator 1 ReadCSV->SplitBlocks(additional=99)" in stats, stats
    assert "Operator 2 MapBatches" in stats, stats

    # Smaller than a single row.
    ds.context.target_max_block_size = 1
    ds = ds.map_batches(lambda x: x).materialize()
    # 150 rows.
    assert ds._logical_plan.initial_num_blocks() == 150
    print(ds.stats())


def test_large_file_additional_split_budget(ray_start_10_cpus_shared, tmp_path):
    ctx = ray.data.context.DataContext.get_current()
    if ctx.use_datasource_v2:
        pytest.skip(
            "V2 defers file listing to execution time, so "
            "``LogicalPlan.initial_num_blocks()`` can't report a "
            "file-count-based estimate pre-materialization. The "
            "post-materialize block-split assertions this test also "
            "makes are still covered by V1."
        )
    ctx.target_max_block_size = 10 * 1024 * 1024

    # ~100MiB of tensor data
    ds = ray.data.range_tensor(1000, shape=(10000,))
    ds.repartition(1).write_parquet(tmp_path)

    ds = ray.data.read_parquet(tmp_path, override_num_blocks=1)
    assert ds._logical_plan.initial_num_blocks() == 1
    print(ds.materialize().stats())
    assert (
        5 < ds.materialize()._logical_plan.initial_num_blocks() < 20
    )  # Size-based block split

    ds = ray.data.read_parquet(tmp_path, override_num_blocks=10)
    assert ds._logical_plan.initial_num_blocks() == 1
    assert 5 < ds.materialize()._logical_plan.initial_num_blocks() < 20

    ds = ray.data.read_parquet(tmp_path, override_num_blocks=100)
    assert ds._logical_plan.initial_num_blocks() == 1
    assert 50 < ds.materialize()._logical_plan.initial_num_blocks() < 200

    ds = ray.data.read_parquet(tmp_path, override_num_blocks=1000)
    assert ds._logical_plan.initial_num_blocks() == 1
    assert 500 < ds.materialize()._logical_plan.initial_num_blocks() < 2000


def test_map_batches_split(ray_start_10_cpus_shared, restore_data_context):
    ds = ray.data.range(1000, override_num_blocks=1).map_batches(
        lambda x: x, batch_size=1000
    )
    assert ds.materialize()._logical_plan.initial_num_blocks() == 1

    ctx = ray.data.context.DataContext.get_current()
    # 100 integer rows per block.
    ctx.target_max_block_size = 800

    ds = ray.data.range(1000, override_num_blocks=1).map_batches(
        lambda x: x, batch_size=1000
    )
    assert ds.materialize()._logical_plan.initial_num_blocks() == 10

    # A single row is already larger than the target block
    # size.
    ds.context.target_max_block_size = 4
    assert ds.materialize()._logical_plan.initial_num_blocks() == 1000


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
