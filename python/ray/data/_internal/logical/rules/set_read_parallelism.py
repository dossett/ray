import logging
from dataclasses import dataclass
from typing import Optional, Union

from ray import available_resources as ray_available_resources
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import (
    AdditionalSplitBudget,
)
from ray.data._internal.logical.interfaces import PhysicalPlan, Rule
from ray.data._internal.logical.operators import Read
from ray.data._internal.util import _autodetect_parallelism
from ray.data.context import WARN_PREFIX, DataContext
from ray.data.datasource.datasource import Datasource, Reader

__all__ = [
    "ReadParallelismDecision",
    "SetReadParallelismRule",
    "compute_read_parallelism",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadParallelismDecision:
    detected_parallelism: int
    reason: str
    estimated_num_blocks: int
    additional_split_budget: Optional[AdditionalSplitBudget]


def compute_read_parallelism(
    datasource_or_legacy_reader: Union[Datasource, Reader],
    parallelism: int,
    mem_size: Optional[int],
    target_max_block_size: Optional[int],
    per_task_row_limit: Optional[int],
    data_context: DataContext,
) -> ReadParallelismDecision:
    """Compute read parallelism and any global additional-split budget."""

    detected_parallelism, reason, _ = _autodetect_parallelism(
        parallelism,
        target_max_block_size,
        data_context,
        datasource_or_legacy_reader,
        mem_size,
    )
    # Match plan_read_op's execution-time arguments to minimize task-count drift.
    # If counts still differ, runtime only grants budget to planned task indices.
    read_tasks = datasource_or_legacy_reader.get_read_tasks(
        detected_parallelism,
        per_task_row_limit=per_task_row_limit,
        data_context=data_context,
    )
    num_read_tasks = len(read_tasks)
    expected_block_size = None
    if mem_size and num_read_tasks > 0:
        expected_block_size = mem_size / num_read_tasks
        logger.debug(
            f"Expected in-memory size {mem_size}, block size {expected_block_size}"
        )
        if target_max_block_size is None:
            # Unlimited block size -> no extra splits
            size_based_splits = 1
        else:
            size_based_splits = round(
                max(1, expected_block_size / target_max_block_size)
            )
    else:
        size_based_splits = 1
    logger.debug(f"Size based split factor {size_based_splits}")
    estimated_num_blocks = num_read_tasks * size_based_splits
    logger.debug(f"Blocks after size splits {estimated_num_blocks}")

    available_cpu_slots = ray_available_resources().get("CPU", 1)
    if (
        parallelism != -1
        and num_read_tasks >= available_cpu_slots * 4
        and num_read_tasks >= 5000
    ):
        logger.warning(
            f"{WARN_PREFIX} The requested number of read blocks of {parallelism} "
            "is more than 4x the number of available CPU slots in the cluster of "
            f"{available_cpu_slots}. This can "
            "lead to slowdowns during the data reading phase due to excessive "
            "task creation. Reduce the value to match with the available "
            "CPU slots in the cluster, or set override_num_blocks to -1 for Ray Data "
            "to automatically determine the number of read tasks blocks."
            "You can ignore this message if the cluster is expected to autoscale."
        )

    # Assign the missing block boundaries across the read tasks if needed.
    # TODO(swang): For parallelism=-1 (user did not explicitly set
    # parallelism), and if the following operator produces much larger blocks,
    # we should scale down the target max block size here instead of using
    # splitting, which can have higher memory usage.
    if estimated_num_blocks < detected_parallelism and estimated_num_blocks > 0:
        additional_split_budget = detected_parallelism - estimated_num_blocks
        estimated_num_blocks += additional_split_budget
        return ReadParallelismDecision(
            detected_parallelism=detected_parallelism,
            reason=reason,
            estimated_num_blocks=estimated_num_blocks,
            additional_split_budget=AdditionalSplitBudget(
                additional_split_budget=additional_split_budget,
                num_tasks=num_read_tasks,
            ),
        )

    return ReadParallelismDecision(
        detected_parallelism=detected_parallelism,
        reason=reason,
        estimated_num_blocks=estimated_num_blocks,
        additional_split_budget=None,
    )


class SetReadParallelismRule(Rule):
    """
    This rule sets the read op's task parallelism based on the target block
    size, the requested parallelism, the number of read files, and the
    available resources in the cluster.

    If the estimated number of natural output blocks is lower than requested,
    this rule distributes the missing block boundaries across the read tasks.
    """

    def apply(self, plan: PhysicalPlan) -> PhysicalPlan:
        ops = [plan.dag]

        while len(ops) > 0:
            op = ops.pop(0)
            if isinstance(op, InputDataBuffer):
                continue
            logical_op = plan.op_map[op]
            if isinstance(logical_op, Read):
                self._apply(plan, op, logical_op)
            ops += op.input_dependencies

        return plan

    def _apply(self, plan: PhysicalPlan, op: PhysicalOperator, logical_op: Read):
        estimated_in_mem_bytes = logical_op.infer_metadata().size_bytes

        decision = compute_read_parallelism(
            logical_op.datasource_or_legacy_reader,
            logical_op.parallelism,
            estimated_in_mem_bytes,
            op.target_max_block_size_override or op.data_context.target_max_block_size,
            per_task_row_limit=logical_op.per_block_limit,
            data_context=op.data_context,
        )

        if logical_op.parallelism == -1:
            assert decision.reason != ""
            logger.debug(
                f"Using autodetected parallelism={decision.detected_parallelism} "
                f"for operator {logical_op.name} to satisfy {decision.reason}."
            )
        plan.op_map[op] = logical_op.set_detected_parallelism(
            decision.detected_parallelism
        )

        if decision.additional_split_budget is not None:
            logger.debug(
                "To satisfy the requested parallelism of "
                f"{decision.detected_parallelism}, "
                "the read tasks are assigned an additional block-boundary budget."
            )
            op.set_additional_split_budget(decision.additional_split_budget)

        logger.debug(f"Estimated num output blocks {decision.estimated_num_blocks}")
