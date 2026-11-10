# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exports trajectory metadata and metrics from a store as a flat DataFrame.

`export_trajectories_df` left-joins each trajectory's `TrajectoryMetadata` with
its `TrajectoryMetrics` on `trajectory_id`, producing one row per trajectory.
Step data is never loaded, so the export stays cheap for large runs.

Nested dictionaries and models are flattened into dot-separated columns:

* `trajectory_id`, then every other metadata field in declaration order, e.g.
  `session_id`, `agent.name`, `final_metrics.total_steps`,
  `final_metrics.extra.<key>`, `extra.<key>` (and, for metadata subclasses,
  their extension fields such as `status` or `total_reward`).
* `metrics.<key>` for every key in `TrajectoryMetrics.extra`.

Trajectories without recorded metrics keep their row, with null `metrics.*`
values. Lists (e.g. `agent.tool_definitions`) are kept as single cells.
"""

from collections.abc import Iterator, Mapping
from typing import Any

import pandas as pd
from tunix.dist_rl.trajectory import base_store
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

TRAJECTORY_ID_COLUMN = "trajectory_id"
METRICS_PREFIX = "metrics"
_SEP = "."


def _flatten(
    value: Mapping[str, Any], prefix: str = ""
) -> Iterator[tuple[str, Any]]:
  """Yields `(dotted_key, leaf)` pairs for the nested mapping `value`.

  Empty mappings yield nothing, so they surface as nulls in the DataFrame.

  Args:
    value: The mapping to flatten.
    prefix: Dotted key prefix prepended to every yielded key.
  """
  for key, item in value.items():
    column = f"{prefix}{key}"
    if isinstance(item, Mapping):
      yield from _flatten(item, prefix=f"{column}{_SEP}")
    else:
      yield column, item


def _drop_placeholder_columns(df: pd.DataFrame) -> pd.DataFrame:
  """Drops all-null columns `c` that also have nested `c.<key>` columns.

  A field that is a dict in some rows and None in others (e.g. `extra`) yields
  both `extra.<key>` columns and an all-null `extra` column; the latter carries
  no information.

  Args:
    df: The flattened DataFrame.

  Returns:
    `df` without placeholder columns.
  """
  columns = list(df.columns)
  placeholders = [
      column
      for column in columns
      if any(other.startswith(f"{column}{_SEP}") for other in columns)
      and df[column].isna().all()
  ]
  return df.drop(columns=placeholders)


def _get_metrics(
    reader: base_store.TrajectoryReader[Any],
    trajectory_ids: list[str] | None,
) -> dict[str, trajectory_lib.TrajectoryMetrics]:
  """Returns the recorded metrics for `trajectory_ids`, skipping missing ones.

  `get_trajectories_metrics(ids)` raises if any ID has no recorded metrics, so
  this falls back to reading every recorded metric and filtering.

  Args:
    reader: The store to read from.
    trajectory_ids: IDs to read metrics for, or None for all recorded metrics.
  """
  if trajectory_ids is None:
    return reader.get_trajectories_metrics()
  try:
    return reader.get_trajectories_metrics(trajectory_ids)
  except base_store.TrajectoryMetricsNotFoundError:
    wanted = set(trajectory_ids)
    return {
        trajectory_id: metrics
        for trajectory_id, metrics in reader.get_trajectories_metrics().items()
        if trajectory_id in wanted
    }


def export_trajectories_df(
    reader: base_store.TrajectoryReader[Any],
    trajectory_ids: list[str] | None = None,
) -> pd.DataFrame:
  """Returns one flat row per trajectory, combining its metadata and metrics.

  Rows follow the order returned by `reader.get_trajectories_metadata` (the
  order of `trajectory_ids` if given). This is a left join on `trajectory_id`:
  trajectories with metadata but no metrics have null `metrics.*` values, and
  metrics recorded for trajectories without metadata are omitted.

  Args:
    reader: The store to export from.
    trajectory_ids: Optional IDs to export. If None, exports every trajectory
      with metadata in the store.

  Returns:
    A DataFrame with a `trajectory_id` column, one dot-separated column per
    flattened metadata field, and one `metrics.<key>` column per key in
    `TrajectoryMetrics.extra`. An empty store yields an empty DataFrame with
    only the `trajectory_id` column.

  Raises:
    TrajectoryMetadataNotFoundError: If any requested trajectory ID has no
      metadata.
    ValueError: If a flattened metadata column collides with the reserved
      `metrics` prefix.
  """
  metadata_rows = [
      dict(_flatten(metadata.model_dump()))
      for metadata in reader.get_trajectories_metadata(trajectory_ids)
  ]
  if not metadata_rows:
    return pd.DataFrame(columns=[TRAJECTORY_ID_COLUMN])
  metadata_df = _drop_placeholder_columns(pd.DataFrame(metadata_rows))
  metadata_df.insert(
      0, TRAJECTORY_ID_COLUMN, metadata_df.pop(TRAJECTORY_ID_COLUMN)
  )

  collisions = [
      column
      for column in metadata_df.columns
      if column == METRICS_PREFIX
      or column.startswith(f"{METRICS_PREFIX}{_SEP}")
  ]
  if collisions:
    raise ValueError(
        f"Metadata columns {collisions} collide with the reserved"
        f" {METRICS_PREFIX!r} column prefix."
    )

  metrics_rows = [
      {
          TRAJECTORY_ID_COLUMN: trajectory_id,
          **dict(
              _flatten(metrics.extra or {}, prefix=f"{METRICS_PREFIX}{_SEP}")
          ),
      }
      for trajectory_id, metrics in _get_metrics(reader, trajectory_ids).items()
  ]
  if not metrics_rows:
    return metadata_df
  metrics_df = _drop_placeholder_columns(pd.DataFrame(metrics_rows))
  return metadata_df.merge(
      metrics_df, how="left", on=TRAJECTORY_ID_COLUMN, validate="many_to_one"
  )
