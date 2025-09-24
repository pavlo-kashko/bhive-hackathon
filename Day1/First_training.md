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
