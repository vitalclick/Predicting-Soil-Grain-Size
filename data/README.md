# Data

The competition data is not redistributed here. Download it from the
[competition page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/data) (requires accepting the
competition rules), or with the Kaggle CLI:

```bash
kaggle competitions download -c soil-grain-size-from-photos
unzip soil-grain-size-from-photos.zip -d data/
```

Expected layout (the same as `/kaggle/input/soil-grain-size-from-photos`):

```
data/
├── Training_labels_updated.csv
├── ppm_updated.csv
├── sample_submission.csv
├── Training-All_Photos_updated/Training-All_Photos_updated/*.jpg
└── Test_All_Photos/Test_All_Photos/*.JPG
```
