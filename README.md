# TopicAnalyser

This repository provides tools for topic modeling and topic extraction using Latent Dirichlet Allocation (LDA). The project includes notebooks and scripts to preprocess data, train models, and analyze topics.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Notebooks](#notebooks)
  - [Scripts](#scripts)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

TopicAnalyser is designed to help you understand the underlying topics in a large collection of documents. This can be useful for content recommendation, document classification, trend analysis, and more.

## Installation

To get started, clone the repository and install the dependencies using Poetry:

```bash
git clone https://github.com/yourusername/TopicAnalyser.git
cd TopicAnalyser
poetry install
```

Make sure you have Python 3.11 installed. Activate the virtual enviroment.

## Usage

### Notebooks

The repository includes several Jupyter notebooks for different stages of topic modeling:
1. **01_data_preparation.ipynb**: Load and prepare text data for topic modeling.
2. **02_topic_modelling.ipynb**: Preprocess data and train an LDA model.
3. **03_check_convergence.ipynb**: Evaluate model convergence using loglikelihood.
4. **04_coherence_evaluation.ipynb**: Evaluate topic coherence.
5. **05_topic_analyser.ipynb**: Analyze topics, visualize top words, and interpret topics.

### Scripts

Additionally, you can run multiple models using the `topic_modeller.py` script as mentioned in the `model_training` notebook. This allows for batch processing of different configurations and datasets. Use the following command to run the script with a specific configuration file:

```bash
python src/topic_modeller.py -c configs/model_configs.jsonl     
```

All models will be saved in the `/models` folder.

## Configuration

Configuration files (`model_configs.jsonl`) are used to set parameters for the scripts. Each configuration file should contain a list of JSON objects, where each object specifies the parameters for a single model run. The `dataset_type` parameter should be either "HF" (Hugging Face) or "Local". The `dataset_name` parameter should be a dataset name from Hugging Face Datasets or a local file name.

Example configuration:

```json
[
    {
        "dataset_type": "HF",
        "dataset_name": "cohere/movies",
        "n_topics": 10,
        "alpha": 0.1,
        "eta": 0.01,
        "run_id":1
    },
    {
        "dataset_type": "Local",
        "dataset_name": "data/my_local_dataset.csv",
        "n_topics": 15,
        "alpha": 0.2,
        "eta": 0.02,
        "run_id":2
    }
]
```

### OpenAI Integration

For topic analysis with OpenAI models, you need to set up your Azure OpenAI credentials. Copy the `.env_template` file to `.env` and fill in the required variables with your Azure OpenAI details:

```bash
cp .env_template .env
```

Edit the `.env` file to include your Azure OpenAI credentials.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
