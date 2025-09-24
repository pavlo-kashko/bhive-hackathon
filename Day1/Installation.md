# bhive-hackathon

b-hive is a framework for Machine Learning tasks on flat root files. It aims to provide full pipelines form dataset export to evaluation with state-of-the-art taggers, like ParticleNet and ParticleTransformer. For questions, please contact the [b-hive-contact](ulrich.willemsen@cern.ch).
### Join the [Mattermost-Channel!](https://mattermost.web.cern.ch/cms-exp/pl/ft4fewfa4b86t8z8nsza7taq1a)


## b-hive

Developing and deploying jet/event tagging algorithms in the realm of high-energy physics (HEP) can be a complex endeavor, often hindered by application-specific obstacles such as processing large datasets in `root` format, efficient ML specific data provisioning, and accurate performance evaluation. To address these challenges, the **b-hive** framework provides a streamlined, modular, and collaborative environment designed to simplify and enhance the workflow of HEP researchers. 

A key feature of **b-hive** is that its top-level code is agnostic to specific applications. Instead of embedding details directly, **b-hive** uses modules such as:

- **Model Type:** Neural network architectures or ML frameworks like `PyTorch`.
- **Model Parameters:** Hyperparameters like learning rate, batch size, optimizer, etc.
- **Physics Selection:** Event selections or `coffea` processes.
- **Kinematic Phase Space Selection:** e.g. $p_T > 30\,\mathrm{GeV}$ or angular distributions in terms of $\eta$.


## Setup

The codebase is python based, using tools like [awkward](https://awkward-array.org/doc/main/), [uproot](https://uproot.readthedocs.io/en/latest/) for reading in the root-files and [numpy](https://numpy.org), [PyTorch](https://pytorch.org) for the training. The different tasks are stitched together with the workflow management system [law](https://law.readthedocs.io/en/latest/).

### Microforge3 (optional)

We will need space for the environment (~10Gb), framework and training data, in general ~50Gb. On lxplus it's advised to use `/eos` (max 1TB) instead of limited `/afs` (max 10GB).
For installation of environment we recommend using mamba. If you don't have it you may install miniforge3 (more details [here]([url](https://github.com/conda-forge/miniforge))):

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### Cloning repository

Clone the repository via gitlab. You may use either ssh or https. Also you will be rewuired to enter your CERN credentials (username and password):

```bash
# clone the repository
git clone https://git@gitlab.cern.ch:7999/cms-btv/b-hive.git
cd b-hive
```

### Setting environment

Next, install the needed python environment via [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) (or conda). This will take around :

```bash
mamba env create -n b_hive -f env.yml
mamba activate b_hive
```

Next, some environment variables need to be configured. These should be put inside the `local_setup.sh`, which is ignored by git, but used by the `setup.sh`. The variables `$DATA_PATH` and `$TESTDIRECTORY` specifies, where the outputs and test files will be written to (we recommend folder output inside b-hive repository). However remember that you need sufficient space!

```bash
# example local_setup.sh
export DATA_PATH=/YOUR/PATH/b-hive/output/
export LXUSERNAME=cern_username
export TESTDIRECTORY=/YOUR/PATH/b-hive/tests/
```

Now source the setup-script to set up the environment everything:

```bash
# initialize needed environments 
source setup.sh
# init law tasks (needed to be executed only once!)
law index
```

## Tutorial

The preparation and utilization of the jet/event tagging models required several main steps defined as tasks in the **b-hive** architecture (see Fig. [bhive_structure]):

- **DatasetConstructorTask**: creating a prepared and cleaned dataset in compressed format from Monte Carlo simulation typically saved in `root` files.
- **TrainingTask**: the direct updating of the model's parameters using the prepared dataset in order to improve accuracy and decrease loss.
- **InferenceTask**: the evaluation of the trained model's performance, typically on the independent dataset (an additional DatasetConstructorTask should be run here or will be requested by the `law` pipeline).
- **ROCCurveTask**: using predictions of the classes from the inference step to plot the Receiver Operating Characteristic (ROC) curve, showing the background rejection for different signal efficiencies.


The orchestration of tasks within **b-hive** is managed by the **LAW** (Luigi Analysis Workflow) Python package [[law]](https://law.readthedocs.io/en/latest/#).


### List tasks

To get familiar with the structure, it is advisable to list the available tasks:

```bash
# list all tasks
law index --verbose
"""
output
"""
indexing tasks in 8 module(s)
loading module 'tasks.base', done
loading module 'tasks.dataset', done
loading module 'tasks.plotting', done
loading module 'tasks.training', done
loading module 'tasks.inference', done
loading module 'tasks.working_point', done
loading module 'tasks.input_plotter', done
loading module 'workflows.DeepFlavour', done

module 'workflows.DeepFlavour', 4 task(s):
    - DeepJetRunHLT
    - DeepJetRun
    - DeepJetRunHLTPhase2
    - ParticleNetRunHLT

module 'tasks.dataset', 1 task(s):
    - DatasetConstructorTask

module 'tasks.training', 1 task(s):
    - TrainingTask

module 'tasks.inference', 1 task(s):
    - InferenceTask

module 'tasks.plotting', 1 task(s):
    - ROCCurveTask

module 'tasks.working_point', 1 task(s):
    - WorkingPointTask

module 'tasks.input_plotter', 2 task(s):
    - InputHistogrammerTask
    - HistogramPlotterTask

written 11 task(s) to index file '/path/to/folder/b-hive/.law/index'

```

This shows us all taks that are at hand:

- DatasetConstructorTask
    - exports a dataset of root file to the training format
- TrainingTask 
    -  trains a model
- InferenceTask
    - runs the prediction on a dataset
- ROCCurveTask 
    - computes and plots the ROC on a prediction
- WorkingPointTask
    - calculates working points on a prediction
- InputHistogrammerTask
    - creates histograms of input features
- HistogramPlotterTask
    - plot histograms of input features

all these need to be in files that are specified in the `law.cfg`. Task that are in the `tasks` directory, are basic tasks, whereas the `workflows` directory specifies full training pipelines, with the necessary parameters set.

### DatasetConstructorTask

To learn, what parameters a tasks accepts, you can either have a look into the code, or
type:
```bash
law run DatasetConstructorTask <TAB><TAB>
```
so please press double-tab to list all acceptable parameters
```bash
law run DatasetConstructorTask
--chunk-size
--config
--debug
--help
--log-file
--print-deps
--print-status
--scheduler-host
--filelist
--verbose
--coffea-worker
--dataset-version
--fetch-output
--local-scheduler
--log-level
--print-output
--remove-output
--scheduler-port
--workers
```

or you can try put
```bash
law run DatasetConstructorTask --help
```
to receive an extensive list. Some of these parameters are by law, whereas others are defined in b-hive.

Let's explain some parameters:
 - --config 
   - this specifies the config that should be used, described in [Config](#config)
 - --dataset-version
   - this specifies the version-tag that should be used, for example `v_01`, `test_new_config_1`, ...
- --filelist
   - .txt file with input root files. This should be a file with absolut paths of the root files
   to read in. Streming with xrootd is possible since these files are read in by coffea.
- --coffea-worker
   - number of workers that should be used to parallelize the conversion. On a large machine, this can be easily set to 32 or 64
- --chunk-size
   - This specifies the number of events/jets/datapoints that will be written to the target files.
   With the number of features, you can roughly estimate the resulting filesize. A reasonable size would be 100000.
- --debug
   - activate debug mode, where not the full fileset is processed but only one per process. This is good for checking if the export is working without processing the full fileset.

Many of these paramters, like `chunk-size`, `coffea-worker` have reasonable defaults set, while
the `filelist` must explicitly be set (without, what would we read in anyway?!).
Some of the parameters are *significant* whereas some others are *insiginificant*. This refers to
if a parameter changes the output or doesn't. For example the `--coffea-worker` parameter only changes
the parallelization but not what is done, whereas the `config` might change, what config is read in.

### Task-Tree

Law is used for datapipeline management. It parses arguments and checks if a task in the pipeline
has already run or if it needs to be run prior before the specified task can be executed.
This is useful, if one wants to produce for example a ROC curve or working points on a specified
dataset but the training has not run - so it is run automatically.

To see the dependencies, one runs the command the task with `--print-status -1`, which prints *all* prior tasks. 
The number specifies the depth, whereas -1 denotes everything
```bash
(b-hive) NiclasEich@vispa-portal2(structured_arrays):~/b_tagging/b-hive$ law run ROCCurveTask --dataset-version run3_test_strucuted_array_24 --DatasetConstructorTask-coffea-worker 64 --DatasetConstructorTask-test-filelist /net/scratch/cms/data/btv/2023_08_22/test_files.txt --DatasetConstructorTask-training-filelist /net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt --config hlt_run3 --TrainingTask-n-threads 20 --model-name DeepJetHLT --batch-size 10000 --epochs 2 --training-version test_structured_array_16 --DatasetConstructorTask-chunk-size 1000000 --print-status -1
/home/NiclasEich/b_tagging/b-hive/tasks/training.py:14: UserWarning: Anomaly Detection has been enabled. This mode will increase the runtime and should only be enabled for debugging.
  torch.autograd.detect_anomaly(True)
print task status with max_depth -1 and target_depth 0

0 > ROCCurveTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000)
│     LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/ROCCurveTask/run3_test_strucuted_array_24/.../loss.pdf)
│       absent
│
├──1 > TrainingTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000, loss_weighting=False)
│  │     training_metrics: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../training_metrics.npz)
│  │       absent
│  │     validation_metrics: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../validation_metrics.npz)
│  │       absent
│  │     model: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../model_1.pt)
│  │       absent
│  │     best_model: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/TrainingTask/run3_test_strucuted_array_24/.../best_model.pt)
│  │       absent
│  │
│  └──2 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_file
│         s.txt, chunk_size=1000000)
│           file_list: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/run3_test_strucuted_array_24/processed_files.txt)
│             existent
│           histogram_training: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/run3_test_strucuted_array_24/histogram_training.npy)
│             existent
│           histogram_test: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/DatasetConstructorTask/run3_test_strucuted_array_24/histogram_test.npy)
│             existent
│
├──1 > InferenceTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000)
│  │     output_root: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../output.root)
│  │       absent
│  │     prediction: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../prediction.npy)
│  │       absent
│  │     process: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../process.npy)
│  │       absent
│  │     truth: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../truth.npy)
│  │       absent
│  │     kinematics: LocalFileTarget(fs=local_fs, path=/net/scratch/NiclasEich/BTV/training/InferenceTask/run3_test_strucuted_array_24/.../kinematics.npy)
│  │       absent
│  │
│  ├──2 > TrainingTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_version=test_structured_array_16, epochs=2, model_name=DeepJetHLT, n_threads=4, batch_size=10000, loss_weighting=False)
│  │  │     outputs already checked
│  │  │
│  │  └──3 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_f
│  │         iles.txt, chunk_size=1000000)
│  │           outputs already checked
│  │
│  └──2 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_file
│         s.txt, chunk_size=1000000)
│           outputs already checked
│
└──1 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=run3_test_strucuted_array_24, training_filelist=/net/scratch/cms/data/btv/2023_08_22/training_files_less_ttbar.txt, test_filelist=/net/scratch/cms/data/btv/2023_08_22/test_files.t
       xt, chunk_size=1000000)
         outputs already checked

```


### Config

To receive a modular setup and create the possibility for multiple groups, to collaborate, different configs are
created in the `config` directory.
These denote the different root-branches that are specified to be exported, together with $p_T$ and $\eta$ bins, and other 
configuration possibilities.

All branches that are specified in the config are exported. The different models than

### Running a Training

Test files are saved in `/eos/cms/store/group/phys_btag/b-hive/test_files/data`. You can download these and then run a training.
The HLT test-files should be used with the `hlt_run3` config, whereas offline may utilize `part_run3_lt`.

Download the files via:
```bash
scp -r user@lxplus.cern.ch:/eos/cms/store/group/phys_btag/b-hive/test_files/data/hlt /YOUR/FOLDER/hlt_test
```
or if you work on lxplus simply:
```bash
cp -r /eos/cms/store/group/phys_btag/b-hive/test_files/data/hlt /YOUR/FOLDER/hlt_test
```

these should give you these three files:
```
out_QCD_2000_test.root  out_TT_11.root  out_TT_1.root
```

now we need to create two filelist with a `training_files.txt` and `test_files.txt`.
This can be either done manually, or with a simple bash command:

```bash
# find all files - works also in nested dirs
find ~+ -type f -name "*.root" > all_files.txt

# put the first 2 files into training
head -n 2 all_files.txt > training_files.txt
# put last 1 into test
tail -n 1 all_files.txt > test_files.txt

# Let's check what we have created here
echo "Training files:"
cat training_files.txt
echo "Test files:"
cat test_files.txt 
```

Now we can run a Dataset-Construction (should take ~2min):
```bash
law run DatasetConstructorTask --dataset-version tutorial_01 --filelist /YOUR/FOLDER/training_files.txt --chunk-size 30000 --config hlt_run3
```

The logs inform you about number of root files in your dataset, used custom coffea processor, number of created compressed training files and the spent time:

```bash
Dataset construction
Number of files: 2
Processor: LZ4Processing
  Preprocessing 100% ━━━━━━━━━━━━━━━━━━ 2/2 [ 0:00:00 < 0:00:00 | 15.5   file/s ]
Merging (local) 100% ━━━━━━━━━━━━━━━━━━ 2/2 [ 0:00:00 < 0:00:00 | ?    merges/s ]
     Processing 100% ━━━━━━━━━━━━━━━━━━ 180/180 [ 0:01:55 < 0:00:00 | 1.5  chunk/s ]
Merging (local) 100% ━━━━━━━━━━━━━━━━━━ 180/180 [ 0:01:55 < 0:00:00 | ?   merges/s ]
Start merging files
/YOUR/PATH/b-hive/output/DatasetConstructorTask/hlt_run3/tutorial_01/file_0.lz4
/YOUR/PATH/b-hive/output/DatasetConstructorTask/hlt_run3/tutorial_01/file_1.lz4
/YOUR/PATH/b-hive/output/DatasetConstructorTask/hlt_run3/tutorial_01/file_2.lz4
Merging... 0:00:11  ━━━━━━━━━━━━━━━━━━ 100% 0:00:00 180/180 its
```

In the end of any `law` proces you should receive the message:
```bash
This progress looks :) because there were no failed tasks or missing dependencies
```

The DatasetConstructorTask should run without error. To check the outputs, we can run:
```bash
law run DatasetConstructorTask --dataset-version tutorial_01 --config hlt_run3 --print-status 0
```

The output will provide us with the information whether the final files have been created and what are their paths:
```bash
print task status with max_depth 0 and target_depth 0

0 > DatasetConstructorTask(debug=False, config=hlt_run3, verbose=False, dataset_version=tutorial_01, chunk_size=1000000)
      file_list: LocalFileTarget(fs=local_fs, path=/YOUR_PATH/b-hive/output/DatasetConstructorTask/hlt_run3/tutorial_01/processed_files.txt)
        existent
      histogram_training: LocalFileTarget(fs=local_fs, path=/YOUR_PATH/b-hive/output/DatasetConstructorTask/hlt_run3/tutorial_01/histogram_training.npy)
        existent
      histogram_test: LocalFileTarget(fs=local_fs, path=/YOUR_PATH/b-hive/output/DatasetConstructorTask/hlt_run3/tutorial_01/histogram_test.npy)
        existent
las run Dataset
```

Great, so we see that we have created the files. You can also call a `ls /YOUR_PATH/b-hive/output/DatasetConstructorTask/hlt_run3/tutorial_01` to see
all the files that are in there:
```bash
file_0.lz4  file_1.lz4  file_2.lz4  histogram.npy  processed_files.txt  weights.json
```
Here `file_{i}.lz4` are the compressed files with jets' information, `processed_files.txt` contains the list of created lz4 files and `weights.json` saves metadata of chunksizes and sum of weights per file which help to economy time in future. 

Now we are ready to run a training. The minimal rewuired information is the training version, training dataset, configuration file, model name and number of epochs (there are much more parameters, check them all with `--help`):
```bash
law run TrainingTask --training-version tutorial_training_01 --dataset-version tutorial_01 --config hlt_run3 --model-name DeepJetHLT --epochs 3
```
The logs contain a lot of useful information. Let's go through this.

Firstly there are law logs, telling that Dataset is already created so we can start TrainingTask with indicated and default values of parameters.
```bash
INFO: luigi-interface - Informed scheduler that task   TrainingTask_nominal_False_1_cc949942ba   has status   PENDING
INFO: luigi-interface - Informed scheduler that task   DatasetConstructorTask_100000_hlt_run3_tutorial_01_b65f628513   has status   DONE
INFO: luigi-interface - Done scheduling tasks
INFO: luigi-interface - Running Worker with 1 processes
INFO: luigi-interface - [pid 4099865] Worker Worker(salt=1770702037, workers=1, host=lxplus9106.cern.ch, username=pkashko, pid=4099865) running   TrainingTask(debug=False, terminal_plot=False, config=hlt_run3, verbose=False, seed=123456, dataset_version=tutorial_01, training_precision=float32, training_version=tutorial_training_01, epochs=3, model_name=DeepJetHLT, n_threads=4, batch_size=1024, learning_rate=0.001, optimizer=AdamW, loss_function=CrossEntropyLoss, betas=0.95,0.999, eps=1e-06, lr_scheduler=epoch_lin_decay, lr_decay_factor=0.01, mixed_precision=False, use_torch_compile=False, torch_compile_mode=default, attack=nominal, attack_magnitude=0.0, attack_iterations=1, attack_individual_factors=False, attack_reduce=True, attack_restrict_impact=-1.0, attack_overshoot=0.02, attack_uncertainty=1.0, loss_weighting=False, resume_training=False, resume_epoch=False, extend_training=0, train_val_split=0.85, n_train_files=-1)
```

Then you have information about training data. The files are divided for training and validation. You also see number of jets in each file:
```bash
Loading Dataset
#Train files: 2
#Val files: 1
chunk_sizes: [30000 30000 19499]
```

As we have unequal number of jet classes in our data we need to deal with it, otherwise the performance will be poor. This can be done in 2 ways either loss or reference class reweighting. The last one is the default, so you have estimated number of jets for training and validation (some variation may have place to the randomness of the selection procees).
```bash
Using weighted sampling
Estimated number of jets/events in training (after reweighing):   22,528
Estimated number of jets/events in validation (after reweighing): 7,168
```

The rest technical details regarding advesarial attacks, name and size of the model etc.:
```bash
Applying nominal attack (attack_magnitude=0.0, n_iterations=1, attack_uncertainty=1.0).
Dataset construction
Precision of training dataset is set to float32
Precision of validation dataset is set to float32
Model:     DeepJetHLT
Optimizer: AdamW
Scheduler: epoch_lin_decay
Total number of parameters: 251,722
Start training on cpu
```

Finally you have live moving logs of the training process (your numbers may be slightly different).
```bash
Start training on cpu
Epoch 1 of 3
entering traing loop
Training...   | Average loss: 1.5810 | Batch loss: 1.2451 | lr: 0.00100 0:02:47 ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 Speed: 0.1 b/s | 22/~22 its
   Average loss: 1.5810
   Average accuracy: 46.6220
Validation... | Average loss: 1.3977 | Batch loss: 1.4296 0:00:14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 0.44 b/s | 7/~7 its
   Validation loss: 1.3977
   Validation accuracy: 67.2352
Epoch 2 of 3
entering traing loop
Training...   | Average loss: 1.1716 | Batch loss: 1.1285 | lr: 0.00100 0:00:14 ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 Speed: 1.5 b/s | 22/~22 its
   Average loss: 1.1716
   Average accuracy: 68.5236
Validation... | Average loss: 1.0556 | Batch loss: 1.0221 0:00:01 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 6.25 b/s | 7/~7 its
   Validation loss: 1.0556
   Validation accuracy: 71.4304
Epoch 3 of 3
entering traing loop
Training...   | Average loss: 1.0606 | Batch loss: 1.0251 | lr: 0.00001 0:00:08 ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 Speed: 2.8 b/s | 22/~22 its
   Average loss: 1.0606
   Average accuracy: 70.0950
Validation... | Average loss: 1.0671 | Batch loss: 1.0546 0:00:01 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 3.43 b/s | 7/~7 its
   Validation loss: 1.0671
   Validation accuracy: 71.3378
Training finished.
```
I hope you are as excited as me to see the loss and accuracy improve!

Finally let's plot some ROC curves to check the performance of our NN classifier. FOr that purpose we should indicate the test dataset:
```bash
law run ROCCurveTask --training-version tutorial_training_01 --dataset-version tutorial_01 --test-filelist /YOUR/FOLDER/test_files.txt --test-dataset-version tutorial_test_files_01 --config hlt_run3 --model-name DeepJetHLT --epochs 3
```

Now if you check prior, which tasks already ran, you will realise that we have not run a prediciton! This is however not a problem, since law recognizes this and rund the tasks itself: firstly the DatasetConstructorTask to create test dataset, then the InferenceTask the run the trained model on test data and finally ROCCurveTask to produce plots.
```bash
INFO: luigi-interface - Informed scheduler that task   ROCCurveTask_nominal_False_1_355aa54dfa   has status   PENDING
INFO: luigi-interface - Informed scheduler that task   DatasetConstructorTask_100000_hlt_run3_tutorial_test_fi_31c2427013   has status   PENDING
INFO: luigi-interface - Informed scheduler that task   InferenceTask_nominal_False_1_355aa54dfa   has status   PENDING
INFO: luigi-interface - Informed scheduler that task   TrainingTask_nominal_False_1_5f644902fa   has status   DONE
INFO: luigi-interface - Done scheduling tasks
```

Now you can have a look at the results located in the indicated paths and maybe start training your tagger on a real dataset! ;)


## Collaborating

Classifiers are based on the parent class `utils/models/base_model.py`. Each model `utils/models` is a child class with its own specification regarding `forward()` function. Other methods could be directly overwritten in the child class, but don't change `base_model.py`.
