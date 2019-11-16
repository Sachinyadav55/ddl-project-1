## Dataloader

Assume the structure of data directories is the following:

```misc
./
  MLclips/
    training/
      class1/ (directories of class names)
        videoName.mp4 (mp4 files)
        ...
      .../
    testing/
      class1/ (directories of class names)
        videoName.mp4 (mp4 files)
        ...
      .../
```

The dataloader will take root path (MLclips) and return iteratble object after applying transformations (under development) on the data