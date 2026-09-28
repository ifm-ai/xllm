from typing import Any, Dict, Iterator, Optional

from xllm.data.dataset_streamer.data_iterator.data_iterator import DataIterator
from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.data.dataset_streamer.feature_builder.feature_builder import FeatureBuilder
from xllm.data.data_types import Instance

# -------------------------
# DatasetStreamer
# -------------------------
class DatasetStreamer:
    def __init__(
        self,
        dataset_dir: str,
        data_iterator: DataIterator,
        tokenizer: Tokenizer,
        feature_builder: FeatureBuilder,
    ):
        self.dataset_dir = dataset_dir
        self.data_iterator = data_iterator
        self.tokenizer = tokenizer
        self.feature_builder = feature_builder

    def start(self) -> None:
        self.data_iterator.start()

    def close(self) -> None:
        self.data_iterator.close()

    def build_features(self, instance: Instance) -> Optional[Instance]:
        try:
            return self.feature_builder.build(instance, self.tokenizer)
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Failed to build features for {instance.filename}:{instance.line_num} "
                f"for dataset {self.dataset_dir!r}: {error}"
            ) from error

    def __iter__(self) -> Iterator[Instance]:
        for instance in self.iter_raw():
            built = self.build_features(instance)
            if built is not None:
                yield built

    def iter_raw(self) -> Iterator[Instance]:
        self.data_iterator.start()
        while True:
            try:
                instance = next(self.data_iterator)
            except StopIteration:
                return
            yield instance

    def get_state(self) -> Dict[str, Any]:
        # Only data_iterator is stateful
        return {
            "data_iterator": self.data_iterator.get_state(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.data_iterator.set_state(state.get("data_iterator", {}))
