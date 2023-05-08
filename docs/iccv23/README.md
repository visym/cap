# Overview

The [Second Workshop on Fine-grained Activity Detection](https://openfad.nist.gov/workshop/iccv_fgad23) will be held on October 2 2023 at [ICCV](https://iccv2023.thecvf.com/).

The workshop will host a new challenge on fine-grained activity recognition.  The competition dataset will be the [Consented Activities of People (CAP)](https://github.com/visym/cap) dataset.  This is a new annotated dataset of 1.45M videos of 512 fine grained activity classes of consented people, curated using the [Collector](https://visym.com/collector) platform.  The Collector platform provides the framework to collect ethically sourced video dataset people performing rare activities in public and shared private spaces.  The dataset is annotated with bounding boxes around the primary actor, and temporal start/end frames for each activity instance.   All videos are collected with informed consent from subjects for how their data will be used, stored and shared.  Videos are submitted by workers around the world, using a mobile-app, providing an ethical platform for on-demand visual dataset collection.

This workshop features an open leaderboard on Fine-grained activity detection.  The [leaderboard](https://openfad.nist.gov/#pills-leaderboard) where we will evaluate activity classification and activity detection in the [CAP dataset](https://github.com/visym/cap).


## Tasks

The tasks for evaluation on this challenge are:

* Activity classification (AC).  The Activity Classification (AC) task is to assign one or more activity class labels and confidence scores to each video clip from a set of predefined classes.  The metric for AC performance is Mean Average Precision (mAP), top-1 and top-5 classification performance averaged over all classes.   Sequestered evaluation data includes 4x more data not previously evaluated in the [First FGAD'23](https://openfad.nist.gov/workshop/fgad23) challenge.

* Temporal Activity Detection (TAD)}.  The Temporal Activity Detection (AD) task is to detect and temporally localize all activity instances in untrimmed video.  The metric for AD performance is Mean Average Precision (mAP) at a fixed temporal intersection over union (IoU) of 0.2, 0.5, 0.8.

The leaderboard evaluation will be performed on a video test set available to the challenge participants, with sequestered ground truth.  Training and validation sets are publicly available.  Test sets are sequestered and are available to the challenge participants for video download following registration and submission of a license agreement.  Challenge participants are required to upload a CSV file with results on each test set videos to an evaluation server, which will then perform the evaluation and push results to the public leaderboard.  A comprehensive [evaluation plan](https://openfad.nist.gov/uassets/3) is available for review.


## Download

* [cap_detection_handheld_val.tar.gz](https://dl.dropboxusercontent.com/s/db11zv0gcycu384/cap_detection_handheld_val.tar.gz.72f58e69582c17dd366d3c7e85cf0da8) (0.9 GB)&nbsp;&nbsp;MD5:72f58e69582c17dd366d3c7e85cf0da8&nbsp;&nbsp;(05May23)
    * Validation set for handheld activity detection in untrimmed clips
* [cap_classification_clip.tar.gz](https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_clip.tar.gz) (288 GB)&nbsp;&nbsp;MD5:54315e2ce204f0dbbe298490a63b5b3b&nbsp;&nbsp;(02Mar22)
    * Tight temporal clip training/validation set for handheld activity classification
* [cap_classification_pad.tar.gz](https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_pad.tar.gz) (386 GB)&nbsp;&nbsp;MD5:fbdc75e6ef10b874ddda20ee9765a710&nbsp;&nbsp;(02Mar22)
    * Temporally padded (&gt;4s) training/validation set for handheld activity classification


# Getting Started

The CAP dataset contains 1.45M clips of 512 fine-grained activity labels.  A performer is required to use the training set of trimmed clips of activities to train a system for activity classification.  This trained activity classification system is then used for untrimmed activity detection.

The download section provides two activity classification training sets.  The first training set contains examples of each activity class with no temporal padding.  The second (optional) training set contains examples of each activity class with equal temporal padding so that each clip has length greater than or equal to 4 seconds.  The training sets are equivalent other than this padding, and a performer may choose which training set to use.

The download section also provides one activity detection validation set.  Unlike the activity classification task, we do not provide an activity detection training set.  A performer is expected to train a representation for each activity class from the AC training set, then a system should leverage this trained system for temporal localization.  The validation set includes 47 examples of 45 second long videos containing one or more fine-grained activity labels.  A performer system is required to temporally localize the start and end frames of each detected activity class and report these times along with a confidence.  The validation set includes visualizations of the ground truth in the 47 videos.  The test data may contain any of the 512 activity labels in the AC training set, and it is the responsibility of the performer system to localize any detected instances in these untrimmed clips.

The training and valiation sets are exported in an open JSON format.  The [VIPY Python tools](https://github.com/visym/vipy) can be used for dataset transformation, visualization and training preparation.











