# Movie Facts and Fibs (MF²): A Benchmark for Long Movie Understanding

## Download the MF2 Dataset

To download the MF2 dataset into the `data/` folder, follow these steps:

1. Make sure [Git LFS](https://git-lfs.github.com/) is installed. Then run:

   ```bash
   git lfs install 
   (cd data && git clone https://huggingface.co/datasets/sardinelab/MF2)
    ```

## Setup Instructions

### 1. Create and Activate a Virtual Environment (Optional but Recommended)

It is recommended to use a virtual environment to manage dependencies. You can create one using `venv`:

```bash
python -m venv mf2-env
source mf2-env/bin/activate 
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. FFmpeg

Make sure ffmpeg is installed and the module is loaded. 
If your system does not use modules, you can install FFmpeg via a package manager.

```bash
sudo apt install ffmpeg
module load ffmpeg
```

## Example of Evaluating Models

### 1. Running an Open-Weight Model

To run inference with an open-weight model, use the following command:

```bash
bash run_open_inference.sh <model> <output_dir> <modality> <prompt_template>
```

##### Arguments:

- `<model>`: Name of the model as defined in `models/registry.py`.
- `<output_dir>`: Directory path where results will be saved.
- `<modality>`: Input type to be received during evaluation, with options:
  - `video_only`
  - `transcripts_only`
  - `video_and_transcripts`
  - `video_and_synopsis`
  - `video_transcripts_and_synopsis`
  - `statement_only`
  - `synopsis_only`
- `<prompt_template>`: Prompt style to be used during evaluation, with options:
  - `explanation`
  - `explanation_free`
  - `direct`
  - `direct_free` — *recommended for best results with open-weight models.*


### 2. Running a Closed Model

To run inference with a closed model, use the following command:

```bash
bash run_closed_inference.sh <model> <output_dir> <modality> <prompt_template> <api_key>
```

##### Arguments:

- `<model>`: e.g. gpt-4o or gemini-2.5-pro-preview-03-25.
- `<output_dir>`: Directory path where results will be saved.
- `<modality>`: Input type to be received during evaluation, with options:
  - `video_only`
  - `transcripts_only`
  - `video_and_transcripts`
  - `video_and_synopsis`
  - `video_transcripts_and_synopsis`
  - `statement_only`
  - `synopsis_only`
- `<prompt_template>`: Prompt style to be used during evaluation, with options:
  - `explanation`
  - `explanation_free` — *recommended for best results with closed models.*
  - `direct`
  - `direct_free`
- `<api_key>`: the key to access the closed model.

### 3. Parsing Model Outputs

After running the model, parse the outputs using the following command:

```bash
python parse_model_outputs.py --output_dir <output_dir> --strategy <strategy>
```

##### Arguments:


- `<output_dir>`: Directory containing the results from running the model.
- `<strategy>`: Parsing strategy to apply, with options:
  - `strict`
  - `first-occurrence`
  - `last-occurrence`
  - `strict-w-fallback-first-occurrence`

##### Notes on Strategy Usage:

- For **open-weight models**, `first-occurrence` and `last-occurrence` can be pretty much interchangeably used, as most models will output just the answer due to the selection of the `direct_free` prompt.

- For **closed models**, `last-occurrence` is recommended since the output includes an explanation first before the final result, as the prompt used is `explanation_free`.


## Contact
For questions or support, contact [miguel.moura.ramos@tecnico.ulisboa.pt](mailto:miguel.moura.ramos@tecnico.ulisboa.pt).


## License
CC-BY-NC-SA 4.0 license. 
This dataset is provided for non-commercial use only.