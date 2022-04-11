# Overview

The Consented Activities of People (CAP) dataset is a fine grained activity dataset for visual AI research curated using the [Visym Collector](https://visym.com/collector) platform.  The CAP dataset contains annotated videos of [fine-grained activity classes](https://github.com/visym/cap/blob/main/classification/cap_labels.csv) of consented people.  Videos are annotated with bounding box tracks around the primary actor along with temporal start/end frames for each activity instance, and distributed in [vipy](https://visym.github.io/vipy) json format.  An interactive visualization and video summary is available for review below.  

The CAP dataset was collected with the following goals:

* **Atomic.** Activities have length ≤ 3 seconds and visually grounded (e.g. activities should be unambiguously determined from the pixels).  
* **Non-overlapping.**  All activities are performed independently, and no activities are performed jointly or simultaneously overlapping with other activities (e.g. a subject will not simultaneously perform the “person uses cell phone” activity while performing the “person removes hat” activity). 
* **Person centered.**  All activities are collected from handheld mobile devices at a fixed security perspective (e.g. looking down on a scene from above) and include a single consented person as the primary subject.  Subjects are tasked with performing specific atomic activities, person/object or person/person interactions.  
* **Fine-grained.**  All activities are selected so that there are subtle differences between classes where the activity representation and discrimination is critical for performance, rather than the scene context or object detection.
* **Around the house.**  The collection involves objects, locations and activities that most collectors have easy access to and can easily perform without practice.
* **Ethical.**  All videos are collected with informed consent for how the videos will be shared and used.  Non-consented subjects have their faces blurred. 
* **Worldwide.**  Videos are collected from over 750 collectors in 33 countries. 
* **Large-scale.**  We provide an open and easily downloaded training/validation set suitable for pre-training.  


This dataset will be associated with the [Open Fine-grained Activity Detection Challenge](https://openfad.nist.gov) to be launched in Spring 2022.  We will evaluate activity classification in trimmed clips and temporal activity detection in untrimmed clips.  


## Explorer

<iframe src="https://htmlpreview.github.io/?https://github.com/visym/cap/blob/main/docs/cap_hoverpixel_selector_56K.html" style="width: 1280px; height: 768px; border: 0px;" allowfullscreen></iframe>

## Visualization

<iframe width="1280" height="768" src="https://www.youtube.com/embed/Je91vWjSHpo" title="Fine-grained Activities of Consented People" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Summary

<img src="cap_histogram.png" width="1280" height="100%" />
<img src="cap_summary.png" width="960" style="display:block;margin-left:auto;margin-right:auto;" height="100%" />

## Download

* [cap_classification_clip.tar.gz](https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_clip.tar.gz) (288 GB)&nbsp;&nbsp;MD5:54315e2ce204f0dbbe298490a63b5b3b&nbsp;&nbsp;Last Updated 02Mar22
    * Tight temporal clip training/validation set 
* [cap_classification_pad.tar.gz](https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_pad.tar.gz) (386 GB)&nbsp;&nbsp;MD5:fbdc75e6ef10b874ddda20ee9765a710&nbsp;&nbsp;
    * Temporally padded (&gt;4s) training/validation set

<!--
* cap_classification_pad_stabilized.tar.gz (XXX GB)&nbsp;&nbsp;MD5:XXXX&nbsp;&nbsp;
    * [Background stabilized](https://github.com/visym/vipy/blob/master/vipy/video.py#L3972-L3996) temporally padded (&gt;4s) training/validation set
-->

## License

Creative Commons Attribution 4.0 International [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research.  Non-consented subjects have their faces blurred out.  

## Reference

[arXiv link - coming soon]


# Acknowledgement

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00344. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

We thank the [AWS Open Data Sponsorship Program](https://registry.opendata.aws/visym-cap) for supporting the storage and distribution of this dataset.

# Contact

Visym Labs <a href="mailto:info@visym.com">&lt;info@visym.com&gt;</a>


