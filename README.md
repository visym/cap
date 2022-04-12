# Consented Activities of People (CAP)
Dataset site: [https://visym.github.io/cap](https://visym.github.io/cap)

# Summary

The Consented Activities of People (CAP) dataset is a fine grained activity dataset for visual AI research curated using the Visym Collector platform. The CAP dataset contains annotated videos of fine-grained activity classes of consented people.  Videos are recorded from a security camera like perspective looking down on a scene from above, of subjects performing every day activities.  Videos are annotated with bounding box tracks around the primary actor along with temporal start/end frames for each activity instance, and distributed in vipy json format. An interactive visualization and video summary is available for review below.

The CAP dataset was collected with the following goals:

* Atomic. Activities have length ≤ 3 seconds and visually grounded (e.g. activities should be unambiguously determined from the pixels).
* Non-overlapping. All activities are performed independently, and no activities are performed jointly or simultaneously overlapping with other activities (e.g. a subject will not simultaneously perform the “person uses cell phone” activity while performing the “person takes off hat” activity).
* Person centered. All activities are collected from handheld mobile devices at a fixed security perspective (e.g. looking down on a scene from above) and include a single consented person as the primary subject. Subjects are tasked with performing specific atomic activities, person/object or person/person interactions.
* Fine-grained. All activities are selected so that there are subtle differences between classes where the activity representation and discrimination is critical for performance, rather than the scene context or object detection.
* Around the house. The collection involves objects, locations and activities that most collectors have easy access to and can easily perform without practice.
* Ethical. All videos are collected with informed consent for how the videos will be shared and used. Non-consented subjects have their faces blurred.
* Worldwide. Videos are collected from over 750 collectors in 33 countries.
* Large-scale. We provide an open and easily downloaded training/validation set suitable for pre-training.

This dataset will be associated with the Open Fine-grained Activity Detection Challenge to be launched in Spring 2022. We will evaluate activity classification in trimmed clips and temporal activity detection in untrimmed clips.


# Download

* [cap_classification_clip.tar.gz](https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_clip.tar.gz) (288 GB)&nbsp;&nbsp;MD5:54315e2ce204f0dbbe298490a63b5b3b&nbsp;
    * Tight temporal clip training/validation set 
* [cap_classification_pad.tar.gz](https://consented-activities-of-people.s3.us-west-2.amazonaws.com/train/cap_classification_pad.tar.gz) (386 GB)&nbsp;&nbsp;MD5:fbdc75e6ef10b874ddda20ee9765a710&nbsp;&nbsp;
    * Temporally padded (&gt;4s) training/validation set


# License

Creative Commons Attribution 4.0 International [(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  Every subject in this dataset has consented to their personally identifable information to be shared publicly for the purpose of advancing computer vision research.  Non-consented subjects have their faces blurred out.  

# Acknowledgement

Supported by the Intelligence Advanced Research Projects Activity (IARPA) via Department of Interior/ Interior Business Center (DOI/IBC) contract number D17PC00344. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright annotation thereon. Disclaimer: The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of IARPA, DOI/IBC, or the U.S. Government.

We thank the [AWS Open Data Sponsorship Program](https://registry.opendata.aws/visym-cap) for supporting the storage and distribution of this dataset.


# Contact

Visym Labs <a href="mailto:info@visym.com">&lt;info@visym.com&gt;</a>


