import logging

from .constants.distances_constants import (
    DISTANCE_COORDINATE,
    DISTANCE_COSINE,
    DISTANCE_EUCLIDEAN,
    DISTANCE_WASSERSTEIN,
)


from .constants.models_constants import (
    MODEL_CNN,
    MODEL_RESNET_18,
    MODEL_RESNET_50,
    MODEL_MOBILENET,
    MODEL_AELXNET,
)

from .constants.datasets_constants import (
    DATA_SET_STL_10,
    DATA_SET_CIFAR_10,
    DATA_SET_CIFAR_100,
    DATA_SET_FMNIST,
    DATA_SET_SVHN,
)

from .constants.data_distribution_constants import (
    DATA_DISTRIBUTION_N_20,
    DATA_DISTRIBUTION_N_30,
    DATA_DISTRIBUTION_DIR,
    DATA_DISTRIBUTION_FIX,
)


class Config:
    def __init__(
        self,
        MODEL_TYPE: str,
        DATASET_TYPE: str,
        DATA_DISTRIBUTION_KIND: str,
        DISTANCE_METRIC: str,
        NUMBER_OF_EPOCHS=None,
        SENSITIVITY_PERCENTAGE=None,
        TRAIN_BATCH_SIZE=None,
        TEST_BATCH_SIZE=None,
        TRANSFORM_INPUT_SIZE=None,
        WEIGHT_DECAY=1e-4,
        NUMBER_OF_CLIENTS=10,
        DIRICHLET_BETA=0.1,
        SAVE_BEFORE_AGGREGATION_MODELS=True,
        DO_CLUSTER=True,
        CLUSTERING_PERIOD=6,
        FEDERATED_LEARNING_ROUNDS=6,
        DESIRED_DISTRIBUTION=[
            [2948, 0, 5293, 0, 0, 0, 0, 0, 0, 0],
            [1000, 0, 2330, 0, 0, 0, 0, 0, 0, 0],
            [1000, 0, 5292, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 4249, 3729, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 3729, 0, 2465, 0, 0, 0],
            [0, 0, 0, 3720, 0, 0, 2145, 0, 0, 0],
            [0, 0, 0, 0, 0, 3865, 2864, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1865, 2863, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 5045, 3248],
            [0, 0, 0, 0, 0, 0, 0, 3465, 0, 1329],
        ],
    ):

        self.MODEL_TYPE = self._validate_model_type(MODEL_TYPE)
        self.DATASET_TYPE = self._validate_dataset_type(DATASET_TYPE)
        self.DATA_DISTRIBUTION = self._validate_data_distribution(
            DATA_DISTRIBUTION_KIND
        )
        self.DISTANCE_METRIC = self._set_distance_metric(DISTANCE_METRIC)
        self.NUMBER_OF_EPOCHS = self._set_number_of_epochs(NUMBER_OF_EPOCHS)
        self.SENSITIVITY_PERCENTAGE = self._set_sensitivity_percentage(
            SENSITIVITY_PERCENTAGE
        )

        self.TRAIN_BATCH_SIZE, self.TEST_BATCH_SIZE, self.TRANSFORM_INPUT_SIZE = (
            self._set_transformer(
                TRAIN_BATCH_SIZE,
                TEST_BATCH_SIZE,
                TRANSFORM_INPUT_SIZE,
            )
        )

        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.NUMBER_OF_CLIENTS = NUMBER_OF_CLIENTS
        self.DIRICHLET_BETA = DIRICHLET_BETA
        self.DESIRED_DISTRIBUTION = DESIRED_DISTRIBUTION
        self.SAVE_BEFORE_AGGREGATION_MODELS = SAVE_BEFORE_AGGREGATION_MODELS
        self.DO_CLUSTER = DO_CLUSTER
        self.CLUSTERING_PERIOD = CLUSTERING_PERIOD
        self.FEDERATED_LEARNING_ROUNDS = FEDERATED_LEARNING_ROUNDS

    def get_config(
        self,
    ):
        config_dic = {
            "MODEL_TYPE": self.MODEL_TYPE,
            "DATASET_TYPE": self.DATASET_TYPE,
            "NUMBER_OF_CLASSES": 100 if self.DATASET_TYPE == DATA_SET_CIFAR_100 else 10,
            "PARTITION": self.DATA_DISTRIBUTION,
            "ROUND_EPOCHS": self.NUMBER_OF_EPOCHS,
            "SENSITIVITY_PERCENTAGE": self.SENSITIVITY_PERCENTAGE,
            "TRAIN_BATCH_SIZE": self.TRAIN_BATCH_SIZE,
            "TEST_BATCH_SIZE": self.TEST_BATCH_SIZE,
            "TRANSFORM_INPUT_SIZE": self.TRANSFORM_INPUT_SIZE,
            "LEARNING_RATE": 0.0001 if self.MODEL_TYPE == MODEL_AELXNET else 0.001,
            "WEIGHT_DECAY": self.WEIGHT_DECAY,
            "NUMBER_OF_CLIENTS": self.NUMBER_OF_CLIENTS,
            "DIRICHLET_BETA": self.DIRICHLET_BETA,
            "DESIRED_DISTRIBUTION": self.DESIRED_DISTRIBUTION,
            "SAVE_BEFORE_AGGREGATION_MODELS": self.SAVE_BEFORE_AGGREGATION_MODELS,
            "DO_CLUSTER": self.DO_CLUSTER,
            "CLUSTERING_PERIOD": self.CLUSTERING_PERIOD,
            "FEDERATED_LEARNING_ROUNDS": self.FEDERATED_LEARNING_ROUNDS,
            "DISTANCE_METRIC": self.DISTANCE_METRIC,
        }

        return config_dic

    def _validate_model_type(self, model_type: str) -> str:
        if model_type not in [
            MODEL_CNN,
            MODEL_RESNET_18,
            MODEL_RESNET_50,
            MODEL_MOBILENET,
            MODEL_AELXNET,
        ]:
            raise (f"unsupported model type, {model_type}")

        return model_type

    def _validate_dataset_type(self, dataset_type: str) -> str:
        if dataset_type not in [
            DATA_SET_STL_10,
            DATA_SET_CIFAR_10,
            DATA_SET_CIFAR_100,
            DATA_SET_FMNIST,
            DATA_SET_SVHN,
        ]:
            raise (f"unsupported dataset type, {dataset_type}")

        return dataset_type

    def _validate_data_distribution(self, kind: str) -> str:
        if kind == DATA_DISTRIBUTION_FIX:
            return "noniid-fix"
        elif kind == DATA_DISTRIBUTION_N_20:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label20"
            else:
                return "noniid-#label2"
        elif kind == DATA_DISTRIBUTION_N_30:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label30"
            else:
                return "noniid-#label3"
        elif kind == DATA_DISTRIBUTION_DIR:
            return "noniid-labeldir"
        else:
            raise (f"unsupported data distribution kind, {kind}")

    def _set_distance_metric(self, metric: str) -> str:
        if metric not in [
            DISTANCE_COORDINATE,
            DISTANCE_COSINE,
            DISTANCE_EUCLIDEAN,
            DISTANCE_WASSERSTEIN,
        ]:
            raise (f"unsupported metric type, {metric}")

        return metric

    def _set_number_of_epochs(self, number_of_epochs) -> int:
        if number_of_epochs is not None:
            return number_of_epochs

        if self.MODEL_TYPE == MODEL_AELXNET or self.MODEL_TYPE == MODEL_RESNET_50:
            number_of_epochs = 10
        else:
            number_of_epochs = 1

        logging.warning(
            f"using default value for `NUMBER_OF_EPOCHS` which is {number_of_epochs}"
        )

        return number_of_epochs

    def _set_sensitivity_percentage(self, sensitivity_percentage) -> int:
        if sensitivity_percentage is not None:
            return sensitivity_percentage

        if self.MODEL_TYPE == MODEL_RESNET_50:
            sensitivity_percentage = 20
        else:
            sensitivity_percentage = 10

        logging.warning(
            f"using default value for `SENSITIVITY_PERCENTAGE` which is {sensitivity_percentage}"
        )

        return sensitivity_percentage

    def _set_transformer(
        self,
        train_batch_size,
        test_batch_size,
        transform_input_size,
    ) -> (int, int, int):

        if (
            train_batch_size is not None
            and test_batch_size is not None
            and transform_input_size is not None
        ):
            return train_batch_size, test_batch_size, transform_input_size

        if self.MODEL_TYPE == MODEL_MOBILENET:
            train_batch_size = 64
            test_batch_size = 64
            transform_input_size = 224
        elif self.MODEL_TYPE == MODEL_RESNET_50:
            train_batch_size = 128
            test_batch_size = 128
            transform_input_size = 32
        else:
            train_batch_size = 128
            test_batch_size = 128
            transform_input_size = 128

        logging.warning(
            f"using default value for `TRAIN_BATCH_SIZE` which is {train_batch_size}"
        )
        logging.warning(
            f"using default value for `TEST_BATCH_SIZE` which is {test_batch_size}"
        )
        logging.warning(
            f"using default value for `TRANSFORM_INPUT_SIZE` which is {transform_input_size}"
        )

        return (
            train_batch_size,
            test_batch_size,
            transform_input_size,
        )
