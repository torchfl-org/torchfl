========
Examples
========

``torchfl`` is primarily built for quick prototyping of federated
learning experiments. However, the models, datasets, and abstractions
can also speed up the non-federated learning experiments. In this
section, we will explore examples and usages under both the settings.

Non-Federated Learning
----------------------

The following steps should be followed on a high-level to train a
non-federated learning experiment. We are using the ``EMNIST (MNIST)``
dataset and ``densenet121`` model for this example.

1. **Import the relevant modules.**

    .. code-block:: python

        from torchfl.datamodules.emnist import EMNISTDataModule
        from torchfl.models.wrapper.emnist import MNISTEMNIST

    .. code-block:: python

        import pytorch_lightning as pl
        from pytorch_lightning.loggers import TensorBoardLogger
        from pytorch_lightning.callbacks import (
            ModelCheckpoint,
            LearningRateMonitor,
            DeviceStatsMonitor,
            ModelSummary,
            ProgressBar,
            ...
        )

    For more details, view the full list of PyTorch Lightning `callbacks <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#callback>`_ and `loggers <https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#loggers>`_ on the official website.

2. **Setup the PyTorch Lightning trainer.**

    .. code-block:: python
        :emphasize-lines: 1, 3, 9

        trainer = pl.Trainer(
            ...
            logger=[
                TensorBoardLogger(
                    name=experiment_name,
                    save_dir=os.path.join(checkpoint_save_path, experiment_name),
                )
            ],
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
                DeviceStatsMonitor(),
                ModelSummary(),
                ProgressBar(),
            ],
            ...
        )

    More details about the PyTorch Lightning `Trainer API <https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#>`_ can be found on their official website.

3. **Prepare the dataset using the** :ref:`torchfl.datamodules`.

    .. code-block:: python
        :emphasize-lines: 1

        datamodule = EMNISTDataModule(dataset_name="mnist")
        datamodule.prepare_data()
        datamodule.setup()

4. **Initialize the model using the** :ref:`torchfl.wrapper`.

    .. code-block:: python
        :emphasize-lines: 2, 8

        # check if the model can be loaded from a given checkpoint
        if (checkpoint_load_path) and os.path.isfile(checkpoint_load_path):
            model = MNISTEMNIST(
                "densenet121", "adam", {"lr": 0.001}
            ).load_from_checkpoint(checkpoint_load_path)

        else:
            pl.seed_everything(42)
            model = MNISTEMNIST("densenet121", "adam", {"lr": 0.001})
            trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

5. **Collect the results.**

    .. code-block:: python

        val_result = trainer.test(
            model, test_dataloaders=datamodule.val_dataloader(), verbose=True
        )
        test_result = trainer.test(
            model, test_dataloaders=datamodule.test_dataloader(), verbose=True
        )

6. **Collect the logs.**

   The corresponding files for the experiment (model checkpoints and
   logger metadata) will be stored at ``default_root_dir`` argument
   given to the PyTorch Lightning ``Trainer`` object in Step 2. For
   this experiment, we use the `Tensorboard <https://www.tensorflow.org/tensorboard>`_ logger.
   To view the logs (and related plots and metrics), go to the
   ``default_root_dir`` path and find the Tensorboard log files. Upload
   the files to the Tensorboard Development portal following the `instructions <https://tensorboard.dev/#get-started>`_.
   Once the log files are uploaded, a unique url to your experiment
   will be generated which can be shared with ease! An example can
   be found for `MNIST <https://tensorboard.dev/experiment/Q1tw19FySLSjLN6CW5DaUw/>`_.


7. **More information about loggers.**

  Note that, ``torchfl`` is compatible with all the loggers supported by
  PyTorch Lightning. More information about the PyTorch Lightning loggers
  can be found `here <https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#loggers>`_.

For full non-federated learning example scripts, check the scripts on `GitHub <https://github.com/vivekkhimani/torchfl/tree/master/examples/trainers>`_.


Federated Learning
------------------

The following steps should be followed on a high-level to train a
federated learning experiment.

1. **Prepare a dataset and create federated data shards with iid or non-iid distribution** :ref:`torchfl.datamodules`.

    .. code-block:: python
        :emphasize-lines: 9, 10, 11

        def get_datamodule() -> EMNISTDataModule:
            datamodule: EMNISTDataModule = EMNISTDataModule(
                dataset_name=SUPPORTED_DATASETS_TYPE.MNIST, train_batch_size=10
            )
            datamodule.prepare_data()
            datamodule.setup()
            return datamodule
            
        agent_data_shard_map = get_agent_data_shard_map().federated_iid_dataloader(
            num_workers=fl_params.num_agents,
            workers_batch_size=fl_params.local_train_batch_size,
        )

2. **Initialize the global model, agents, and distribute it to the agents.**

    .. code-block:: python
        :emphasize-lines: 7, 9, 11

        def initialize_agents(
            fl_params: FLParams, agent_data_shard_map: Dict[int, DataLoader]
        ) -> List[V1Agent]:
            """Initialize agents."""
            agents = []
            for agent_id in range(fl_params.num_agents):
                agent = V1Agent(
                    id=agent_id,
                    model=MNISTEMNIST(
                        model_name=EMNIST_MODELS_ENUM.MOBILENETV3SMALL,
                        optimizer_name=OPTIMIZERS_TYPE.ADAM,
                        optimizer_hparams={"lr": 0.001},
                        model_hparams={"pre_trained": True, "feature_extract": True},
                        fl_hparams=fl_params,
                    ),
                    data_shard=agent_data_shard_map[agent_id],
                )
                agents.append(agent)
            return agents

        global_model = MNISTEMNIST(
            model_name=EMNIST_MODELS_ENUM.MOBILENETV3SMALL,
            optimizer_name=OPTIMIZERS_TYPE.ADAM,
            optimizer_hparams={"lr": 0.001},
            model_hparams={"pre_trained": True, "feature_extract": True},
            fl_hparams=fl_params,
        )

        all_agents = initialize_agents(fl_params, agent_data_shard_map)

3. **Initiliaze an FLParam object with the desired FL hyperparameters and pass it on to the Entrypoint object which will abstract the training.**

    .. code-block:: python
        :emphasize-lines: 1, 8

        fl_params = FLParams(
            experiment_name="iid_mnist_fedavg_10_agents_5_sampled_50_epochs_mobilenetv3small_latest",
            num_agents=10,
            global_epochs=10,
            local_epochs=2,
            sampling_ratio=0.5,
        )
        entrypoint = Entrypoint(
            global_model=global_model,
            global_datamodule=get_agent_data_shard_map(),
            fl_hparams=fl_params,
            agents=all_agents,
            aggregator=FedAvgAggregator(all_agents=all_agents),
            sampler=RandomSampler(all_agents=all_agents),
        )
        entrypoint.run()
