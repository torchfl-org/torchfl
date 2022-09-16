<div align="center">
	<img src="https://raw.githubusercontent.com/vivekkhimani/torchfl/master/docs/_static/images/torchfl-github.png" width="960px" height="480px">
</div>
<hr>
<div align="center">
	<a href="https://pypi.org/project/torchfl"><img src="https://img.shields.io/pypi/v/torchfl"></a>
	<a href="https://github.com/vivekkhimani/torchfl"><img src="https://img.shields.io/github/v/release/vivekkhimani/torchfl"></a>
	<a href="https://pypi.org/project/torchfl"><img src="https://img.shields.io/pypi/pyversions/torchfl"></a>
	<a href="https://pypi.org/project/torchfl"><img src="https://img.shields.io/pypi/wheel/torchfl"></a>
	<a href="https://pypi.org/project/torchfl"><img src="https://img.shields.io/pypi/implementation/torchfl"></a>
	<a href="https://github.com/vivekkhimani/torchfl"><img src="https://img.shields.io/github/contributors/vivekkhimani/torchfl"></a>
	<a href="https://github.com/vivekkhimani/torchfl"><img src="https://img.shields.io/github/commit-activity/w/vivekkhimani/torchfl"></a>
	<a href="https://pypi.org/project/torchfl"><img src="https://img.shields.io/pypi/dm/torchfl"></a>
	<a href="https://github.com/vivekkhimani/torchfl"><img src="https://img.shields.io/github/directory-file-count/vivekkhimani/torchfl"></a>
	<a href="https://github.com/vivekkhimani/torchfl"><img src="https://img.shields.io/github/languages/count/vivekkhimani/torchfl"></a>
	<a href="https://github.com/vivekkhimani/torchfl"><img src="https://img.shields.io/github/languages/top/vivekkhimani/torchfl"></a>
	<a href="https://github.com/vivekkhimani/torchfl"><img src="https://img.shields.io/github/search/vivekkhimani/torchfl/torchfl"></a>
	<a href="https://torchfl.readthedocs.io"><img src="https://img.shields.io/readthedocs/torchfl"</a>
	<a href="https://github.com/vivekkhimani/torchfl/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/torchfl"></a>
</div>
<hr>

## Table of Contents

- [Key Features](#features)
- [Installation](#installation)
- [Examples and Usage](#examples-and-usage)
- [Available Models](#available-models)
- [Available Datasets](#available-datasets)
- [Contributing](#contributing)
- [Credits](#credits)

## Features

- Python 3.6+ support. Built using ```torch-1.10.1```, ```torchvision-0.11.2```, and ```pytorch-lightning-1.5.7```.
- Customizable implementations for state-of-the-art deep learning [models](#available-models) which can be trained in federated or non-federated settings.
- Supports finetuning of the pre-trained deep learning models, allowing for faster training using transfer learning.
- PyTorch LightningDataModule wrappers for the most commonly used [datasets](#available-datasets) to reduce the boilerplate code before experiments.
- Built using the bottom-up approach for the datamodules and models which ensures abstractions while allowing for customization.
- Provides implementation of the federated learning (FL) samplers, aggregators, and wrappers, to prototype FL experiments on-the-go.
- Backwards compatible with the PyTorch LightningDataModule, LightningModule, loggers, and DevOps tools.
- More details about the examples and usage can be found [below](#examples-and-usage).
- For more documentation related to the usage, visit - https://torchfl.readthedocs.io/.

## Installation
### Stable Release
As of now, ```torchfl``` is available on PyPI and can be installed using the following command in your terminal:
```
$ pip install torchfl
```
This is the preferred method to install ```torchfl``` with the most stable release.
If you don't have [pip](https://pip.pypa.io/en/stable/) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

### From sources
The sources for ```torchfl``` can be downloaded from the official [Github repository](https://github.com/vivekkhimani/torchfl).
You can either clone the public repository:
```
$ git clone git://github.com/vivekkhimani/torchfl
```
or download the [tarball](https://github.com/vivekkhimani/torchfl/tarball/master):
```
$ curl -OJL https://github.com/vivekkhimani/torchfl/tarball/master
```
Once you have the copy of the source downloaded, you can install it using:
```
$ python setup.py install
```

## Examples and Usage
Although ```torchfl``` is primarily built for quick prototyping of federated learning experiments, the models, datasets, and abstractions can also speed up the non-federated learning experiments. In this section, we will explore examples and usages under both the settings.

### Non-Federated Learning
The following steps should be followed on a high-level to train a non-federated learning experiment. We are using the ```EMNIST (MNIST)``` dataset and ```densenet121``` for this example.

1. Import the relevant modules.
	```python
	from torchfl.datamodules.emnist import EMNISTDataModule
	from torchfl.models.wrapper.emnist import MNISTEMNIST
	```

	```python
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
	```
	For more details, view the full list of PyTorch Lightning [callbacks](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#callback) and [loggers](https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#loggers) on the official website.
2. Setup the PyTorch Lightning trainer.
	```python
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
	```
	More details about the PyTorch Lightning [Trainer API](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#) can be found on their official website.

3. Prepare the dataset using the wrappers provided by ```torchfl.datamodules```.
	```python
	datamodule = EMNISTDataModule(dataset_name="mnist")
	datamodule.prepare_data()
	datamodule.setup()
	```

4. Initialize the model using the wrappers provided by ```torchfl.models.wrappers```.
	```python
	# check if the model can be loaded from a given checkpoint
	if (checkpoint_load_path) and os.path.isfile(checkpoint_load_path):
		model = MNISTEMNIST(
			"densenet121", "adam", {"lr": 0.001}
			).load_from_checkpoint(checkpoint_load_path)
	else:
		pl.seed_everything(42)
		model = MNISTEMNIST("densenet121", "adam", {"lr": 0.001})
		trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
	```

5. Collect the results.
	```python
	val_result = trainer.test(
		model, test_dataloaders=datamodule.val_dataloader(), verbose=True
	)
	test_result = trainer.test(
		model, test_dataloaders=datamodule.test_dataloader(), verbose=True
	)
	```

6. The corresponding files for the experiment (model checkpoints and logger metadata) will be stored at ```default_root_dir``` argument given to the PyTorch Lightning ```Trainer``` object in Step 2. For this experiment, we use the [Tensorboard](https://www.tensorflow.org/tensorboard) logger. To view the logs (and related plots and metrics), go to the ```default_root_dir``` path and find the Tensorboard log files. Upload the files to the Tensorboard Development portal following the instructions [here](https://tensorboard.dev/#get-started). Once the log files are uploaded, a unique url to your experiment will be generated which can be shared with ease! An example can be found [here](https://tensorboard.dev/experiment/Q1tw19FySLSjLN6CW5DaUw/).

7. Note that, ```torchfl``` is compatible with all the loggers supported by PyTorch Lightning. More information about the PyTorch Lightning loggers can be found [here](https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#loggers).

For full non-federated learning example scripts, check [examples/trainers](https://github.com/vivekkhimani/torchfl/tree/master/examples/trainers).

### Federated Learning
The following steps should be followed on a high-level to train a federated learning experiment.

1. FIXME later
2. FIXME later
3. FIXME later

For full federated learning example scripts, check [examples/federated](https://github.com/vivekkhimani/torchfl/tree/master/examples/federated).

## Available Models
For the initial release, ```torchfl``` will only support state-of-the-art computer vision models. The following table summarizes the available models, support for pre-training, and the possibility of feature-extracting. Please note that the models have been tested with all the available datasets. Therefore, the link to the tests will be provided in the next section.

<table>
	<thead>
		<tr>
			<th>Name</th>
			<th>Pre-Training</th>
			<th>Feature Extraction</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/alexnet.py" target="_blank">AlexNet</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/densenet.py#L20" target="_blank">DenseNet121</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/densenet.py#L78" target="_blank">DenseNet161</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/densenet.py#L136" target="_blank">DenseNet169</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/densenet.py#L194" target="_blank">DenseNet201</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/lenet.py" target="_blank">LeNet</a></td>
			<td>:x:</td>
			<td>:x:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/mlp.py" target="_blank">MLP</a></td>
			<td>:x:</td>
			<td>:x:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/mobilenet.py#L23" target="_blank">MobileNetV2</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/mobilenet.py#L78" target="_blank">MobileNetV3Small</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/mobilenet.py#L140" target="_blank">MobileNetV3Large</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L26" target="_blank">ResNet18</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L76" target="_blank">ResNet34</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L125" target="_blank">ResNet50</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L174" target="_blank">ResNet101</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L223" target="_blank">ResNet152</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L272" target="_blank">ResNext50(32x4d)</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L323" target="_blank">ResNext101(32x8d)</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L374" target="_blank">WideResNet(50x2)</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/resnet.py#L425" target="_blank">WideResNet(101x2)</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/shufflenetv2.py#L20" target="_blank">ShuffleNetv2(x0.5)</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/shufflenetv2.py#L74" target="_blank">ShuffleNetv2(x1.0)</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/shufflenetv2.py#L128" target="_blank">ShuffleNetv2(x1.5)</a></td>
			<td>:x:</td>
			<td>:x:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/shufflenetv2.py#L83" target="_blank">ShuffleNetv2(x2.0)</a></td>
			<td>:x:</td>
			<td>:x:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/squeezenet.py#L19" target="_blank">SqueezeNet1.0</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/squeezenet.py#L83" target="_blank">SqueezeNet1.1</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L45" target="_blank">VGG11</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L95" target="_blank">VGG11_BatchNorm</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L145" target="_blank">VGG13</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L195" target="_blank">VGG13_BatchNorm</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L245" target="_blank">VGG16</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L295" target="_blank">VGG16_BatchNorm</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L345" target="_blank">VGG19</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/models/sota/vgg.py#L395" target="_blank">VGG19_BatchNorm</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
		</tr>
	</tbody>
</table>


## Available Datasets
Following datasets have been wrapped inside a ```LightningDataModule``` and made available for the initial release of ```torchfl```. To add a new dataset, check the source code in ```torchfl.datamodules```, add tests, and create a PR with ```Features``` tag.


<table>
	<thead>
		<tr>
			<th>Group</th>
			<th>Datasets</th>
			<th>IID Split</th>
			<th>Non-IID Split</th>
			<th>Datamodules Tests</th>
			<th>Models</th>
			<th>Models Tests</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/datamodules/cifar.py">CIFAR</a></td>
			<td>
				<ul>
					<li><a href="http://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html">CIFAR10</a></li>
					<li><a href="http://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html">CIFAR100</a></li>
				</ul>
			</td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
			<td>
				<a href="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-datamodules.yml"><img src="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-datamodules.yml/badge.svg"></a>
			</td>
			<td>
				<ul>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/cifar/cifar10">CIFAR10</a>
					</li>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/cifar/cifar100">CIFAR100</a>
					</li>
				</ul>
			</td>
			<td>
				<a href="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-models.yml"><img src="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-models.yml/badge.svg"></a>
			</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/datamodules/emnist.py">EMNIST</a></td>
			<td>
				<ul>
					<li><a href="https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html">By Class</a></li>
					<li><a href="https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html">By Merge</a></li>
					<li><a href="https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html">Balanced</a></li>
					<li><a href="https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html">Digits</a></li>
					<li><a href="https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html">Letters</a></li>
					<li><a href="https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html">MNIST</a></li>
				</ul>
			</td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
			<td><a href="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-datamodules.yml"><img src="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-datamodules.yml/badge.svg"></a></td>
			<td>
				<ul>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/emnist/byclass">By Class</a>
					</li>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/emnist/bymerge">By Merge</a>
					</li>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/emnist/balanced">Balanced</a>
					</li>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/emnist/digits">Digits</a>
					</li>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/emnist/letters">Letters</a>
					</li>
					<li>
						<a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/emnist/mnist">MNIST</a>
					</li>
				</ul>
			</td>
			<td>
				<a href="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-models.yml"><img src="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-models.yml/badge.svg"></a>
			</td>
		</tr>
		<tr>
			<td><a href="https://github.com/vivekkhimani/torchfl/blob/master/torchfl/datamodules/fashionmnist.py">FashionMNIST</a></td>
			<td><a href="https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html">FashionMNIST</a></td>
			<td>:white_check_mark:</td>
			<td>:white_check_mark:</td>
			<td><a href="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-datamodules.yml"><img src="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-datamodules.yml/badge.svg"></a></td>
			<td><a href="https://github.com/vivekkhimani/torchfl/tree/master/torchfl/models/core/fashionmnist">FashionMNIST</a></td>
			<td>
				<a href="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-models.yml"><img src="https://github.com/vivekkhimani/torchfl/actions/workflows/pytest-models.yml/badge.svg"></a>
			</td>
	</tbody>
</table>

## Contributing
Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

### Types of Contributions
#### Report Bugs
Report bugs at [https://github.com/vivekkhimani/torchfl/issues](https://github.com/vivekkhimani/torchfl/issues).

If you are reporting a bug, please include:
- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

#### Fix Bugs
Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

#### Implement Features
Look through the GitHub issues for features. Anything tagged with "enhancement", "help wanted", "feature" is open to whoever wants to implement it.

#### Write Documentation
```torchfl``` could always use more documentation, whether as part of the official torchfl docs, in docstrings, or even on the web in blog posts, articles, and such.

#### Submit Feedback
The best way to send feedback is to file an issue at [https://github.com/vivekkhimani/torchfl/issues](https://github.com/vivekkhimani/torchfl/issues).
If you are proposing a feature:
- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

### Get Started
Ready to contribute? Here's how to set up torchfl for local development.
1. Fork the torchfl repo on GitHub.
2. Clone your fork locally:
```
$ git clone git@github.com:<your_username_here>/torchfl.git
```
3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:
```
$ mkvirtualenv torchfl
$ cd torchfl/
$ python setup.py develop
```
4. Create a branch for local development:
```
$ git checkout -b name-of-your-bugfix-or-feature
```
Now you can make your changes locally and maintain them on your own branch.


5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox:
```
tox
```
Run ```tox --help``` to explore more features for tox.


6. Your changes need to pass the existing test cases and add the new ones if required under the ```tests``` directory. Following approaches can be used to run the test cases.
- To run all the test cases.
```
$ coverage run -m pytest tests
```
- To run a specific file containing the test cases.
```
$ coverage run -m pytest <path-to-the-file>
```

7. Commit your changes and push your branch to GitHub:
```
$ git add --all
$ git commit -m "Your detailed description of your changes."
$ git push origin <name-of-your-bugfix-or-feature>
```
8. Submit a pull request through the Github web interface.
9. Once the pull request has been submitted, the continuous integration pipelines on Github Actions will be triggered. Ensure that all of them pass before one of the maintainers can review the request.

### Pull Request Guidelines
Before you submit a pull request, check that it meets these guidelines:
1. The pull request should include tests.
	- Try adding new test cases for new features or enhancements and make changes to the CI pipelines accordingly.
	- Modify the existing tests (if required) for the bug fixes.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in ```README.md```.
3. The pull request should pass all the existing CI pipelines (Github Actions) and the new/modified workflows should be added as required.
4. Please note that the test cases should only be run in the CI pipeline if the direct/indirect dependencies of the tests have changed. Look at the [workflow YAML files](https://github.com/vivekkhimani/torchfl/tree/master/.github/workflows) for more details or reach out to one of the contributors.

### Deploying
A reminder for the maintainers on how to deploy. Make sure all your changes are committed (including an entry in HISTORY.rst). Then run:
```
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

### Credits
We truly appreciate everyone contributing to and helping us build this community! Thanks for all the love and support. A big shoutout to everyone who has contributed to this project.
#### Core Developers
1. <b>Vivek Khimani</b>
	- <b>Email:</b> [vivekkhimani07@gmail.com](mailto:vivekkhimani07@gmail.com)
	- <b>Website:</b> [https://vivekkhimani.github.io](https://vivekkhimani.github.io)
	- <b>Github:</b> [https://github.com/vivekkhimani](https://github.com/vivekkhimani)
	- <b>Buy Me a Coffee:</b> [https://www.buymeacoffee.com/vivekkhimani](https://www.buymeacoffee.com/vivekkhimani)

#### Contributors
None yet. Why not be the first?
