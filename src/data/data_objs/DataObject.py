from abc import ABC, abstractmethod
import tensorflow as tf

class DataObject(ABC):
    @abstractmethod
    def create_final_dataset(self) -> tf.data.Dataset:
        """Returns the final dataset"""